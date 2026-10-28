// ntt_body_impl.tpp -- the WHOLE kernel body, hand-written.
//
// Included from gen/ntt.hpp.  The generated top (gen/ntt.cpp) holds every interface pragma and
// calls body() with every kernel argument; the status registers are references, so this code
// writes halted / error / tx_id itself.  The Python twin, word for word, is ntt_stream_model()
// in ntt.py.
//
// The command-response contract in the code:
//   - everything the kernel computes with arrives on s_in: the header carries nbatch, and the
//     twiddle factors are constants (q is fixed), so nothing is configured over AXI-Lite;
//   - the status is cleared at the start of every activation, and nothing is kept between
//     polynomials, commands or activations;
//   - on an error: set the status and return without reading anything more from s_in.  Every
//     output burst is a whole polynomial, written only after its input is complete, so no burst
//     is ever open when an error is detected.

#include "include/uint32_array_utils.h"
#include "include/ntt_cmd_hdr.h"
#include "include/ntt_resp_hdr.h"
#include "ntt_zetas.h"

namespace ntt_impl {

static const int N = 256;
static const int LOGN = 8;
static const int MAX_BATCH = 16;
static const unsigned int Q = 8380417;     // 2^23 - 2^13 + 1
static const unsigned int CMD_NTT = 1;
static const unsigned int CMD_END = 2;

typedef ap_uint<23> coeff_t;               // a residue in [0, q)

static inline ap_uint<8> code(NttError e) {
#pragma HLS INLINE
    return (ap_uint<8>)static_cast<unsigned int>(e);
}

// a * b mod q, canonical.  The 46-bit product is folded with 2^23 = 2^13 - 1 (mod q): each fold
// replaces hi * 2^23 by hi * 2^13 - hi.  Three folds leave a value below 2q (at most 8454135),
// and one conditional subtraction finishes.  Bounds: r1 < 2^36, r2 < 2^26 + 2^23, r3 < 2q.
static inline coeff_t mulmod(coeff_t a, coeff_t b) {
#pragma HLS INLINE
    const ap_uint<46> p = (ap_uint<46>)a * b;
    const ap_uint<23> p_hi = p.range(45, 23);
    const ap_uint<23> p_lo = p.range(22, 0);
    const ap_uint<37> r1 = (((ap_uint<37>)p_hi << 13) - p_hi) + p_lo;
    const ap_uint<14> r1_hi = r1.range(36, 23);
    const ap_uint<23> r1_lo = r1.range(22, 0);
    const ap_uint<28> r2 = (((ap_uint<28>)r1_hi << 13) - r1_hi) + r1_lo;
    const ap_uint<5> r2_hi = r2.range(27, 23);
    const ap_uint<23> r2_lo = r2.range(22, 0);
    const ap_uint<25> r3 = (((ap_uint<25>)r2_hi << 13) - r2_hi) + r2_lo;
    return (coeff_t)((r3 >= Q) ? (ap_uint<25>)(r3 - Q) : r3);
}

static inline coeff_t addmod(coeff_t a, coeff_t b) {
#pragma HLS INLINE
    const ap_uint<24> s = (ap_uint<24>)a + b;
    return (coeff_t)((s >= Q) ? (ap_uint<24>)(s - Q) : s);
}

static inline coeff_t submod(coeff_t a, coeff_t b) {
#pragma HLS INLINE
    const ap_uint<24> d = (ap_uint<24>)a + Q - b;   // in (0, 2q)
    return (coeff_t)((d >= Q) ? (ap_uint<24>)(d - Q) : d);
}

// One butterfly stage of FIPS 204 Algorithm 41, from `in` to `out` (ping-pong, so the loop
// reads two words of one RAM and writes two of the other: one butterfly per clock).  Stage s has
// len = 128 >> s and 2^s groups; group g uses ZETAS[2^s + g], which is m in Algorithm 41.
static void ntt_stage(const coeff_t in[N], coeff_t out[N], int s) {
#pragma HLS INLINE off
    const int lg = (LOGN - 1) - s;         // log2(len)
    const int len = 1 << lg;
STAGE_BFLY:
    for (int b = 0; b < N / 2; ++b) {
#pragma HLS PIPELINE II=1
        const int grp = b >> lg;
        const int k = b & (len - 1);
        const int j = (grp << (lg + 1)) | k;
        const coeff_t z = (coeff_t)ZETAS[(1 << s) + grp];
        const coeff_t u = in[j];
        const coeff_t t = mulmod(z, in[j + len]);
        out[j] = addmod(u, t);
        out[j + len] = submod(u, t);
    }
}

// The 8 stages: a -> b -> a ... ; after the eighth the result is back in a.
static void ntt_transform(coeff_t a[N], coeff_t b[N]) {
NTT_STAGES:
    for (int s = 0; s < LOGN; s += 2) {
        ntt_stage(a, b, s);
        ntt_stage(b, a, s + 1);
    }
}

// Read one polynomial's burst into a.  Stops at the first bad word, reading nothing after it:
// a word >= q is BAD_COEFF (checked before its TLAST, so it wins on the same word), a TLAST
// before word 255 is TLAST_EARLY.  Word 255 without TLAST is TLAST_MISSING, returned after the
// polynomial is complete.
template <int in_bw>
static ap_uint<8> load_poly(hls::stream<streamutils::axi4s_word<in_bw>>& s_in, coeff_t a[N]) {
#pragma HLS INLINE off
    ap_uint<8> err = code(NttError::NO_ERROR);
LOAD:
    for (int k = 0; k < N; ++k) {
#pragma HLS PIPELINE II=1
        uint32_array_utils::value_type x[1];
        streamutils::tlast_status tl = streamutils::tlast_status::no_tlast;
        uint32_array_utils::read_axi4_stream_lane<in_bw>(s_in, x, 1, tl);
        const bool last = (tl == streamutils::tlast_status::tlast_at_end);
        if (x[0] >= Q) {
            err = code(NttError::BAD_COEFF);
            break;
        }
        a[k] = (coeff_t)x[0];
        if (last && k < N - 1) {
            err = code(NttError::TLAST_EARLY);
            break;
        }
        if (k == N - 1 && !last) {
            err = code(NttError::TLAST_MISSING);
        }
    }
    return err;
}

// Write one transformed polynomial as a closed 256-word burst.
template <int out_bw>
static void store_poly(const coeff_t a[N], hls::stream<streamutils::axi4s_word<out_bw>>& m_out) {
#pragma HLS INLINE off
STORE:
    for (int k = 0; k < N; ++k) {
#pragma HLS PIPELINE II=1
        uint32_array_utils::value_type y[1];
        y[0] = a[k];
        uint32_array_utils::write_axi4_stream_lane<out_bw>(y, m_out, k == N - 1, 1);
    }
}

// The kernel: one activation, a loop over in-band commands until END or an error.
template <int in_bw, int out_bw>
void body(hls::stream<streamutils::axi4s_word<in_bw>>& s_in,
          hls::stream<streamutils::axi4s_word<out_bw>>& m_out,
          ap_uint<1>& halted, ap_uint<8>& error, ap_uint<16>& tx_id) {
#pragma HLS INLINE
    halted = 0;                                   // a clear status for this activation
    error = code(NttError::NO_ERROR);
    tx_id = 0;

    coeff_t a[N];
    coeff_t b[N];
#pragma HLS BIND_STORAGE variable=a type=ram_t2p impl=bram
#pragma HLS BIND_STORAGE variable=b type=ram_t2p impl=bram

CMDS:
    while (true) {
        NttCmdHdr hdr;
        hdr.read_axi4_stream<in_bw>(s_in);        // one word; its TLAST is not inspected
        if (hdr.cmd_type == CMD_END) {
            return;
        }
        ap_uint<8> err = code(NttError::NO_ERROR);
        if (hdr.cmd_type != CMD_NTT) {            // BAD_CMD wins over BAD_PARAM
            err = code(NttError::BAD_CMD);
        } else if (hdr.nbatch == 0 || hdr.nbatch > MAX_BATCH) {
            err = code(NttError::BAD_PARAM);
        }
        if (err == code(NttError::NO_ERROR)) {
            NttRespHdr resp;
            resp.tx_id = hdr.tx_id;
            resp.nbatch = hdr.nbatch;
            resp.write_axi4_stream<out_bw>(m_out, true);
        POLYS:
            for (int p = 0; p < hdr.nbatch; ++p) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=16
                err = load_poly<in_bw>(s_in, a);
                if (err == code(NttError::BAD_COEFF) || err == code(NttError::TLAST_EARLY)) {
                    break;                        // nothing output for this polynomial
                }
                ntt_transform(a, b);
                store_poly<out_bw>(a, m_out);
                if (err != code(NttError::NO_ERROR)) {
                    break;                        // TLAST_MISSING: halt after its output
                }
            }
        }
        if (err != code(NttError::NO_ERROR)) {
            error = err;
            tx_id = hdr.tx_id;
            halted = 1;
            return;
        }
    }
}

}  // namespace ntt_impl
