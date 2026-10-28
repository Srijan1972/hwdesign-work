// ntt_tb.cpp -- the hand-written C++ testbench.
//
// Usage: ntt_tb <data_dir> <stage> [scenario]
//
// For every scenario named in <data_dir>/scenarios.txt (or just [scenario]) it runs the kernel
// once -- one ap_start -- and records what came out:
//
//   1. poison the status registers, so a kernel that does not clear them fails;
//   2. play the stimulus <data_dir>/<scenario>/in into FRESH streams and call the kernel;
//   3. record the response into <data_dir>/<scenario>/<stage>/, and the status in status.json;
//   4. count and discard whatever a halted kernel left unread: the host's reset of the stream
//      path.  The count goes in status.json too, so the checker can confirm the kernel read
//      nothing after an error.
//
// <stage> is "csim" or "cosim".  The output directories must exist; ntt_build.py creates them.
// The stimulus is the same file the Python model reads (ntt.py, ntt_stream_model), and
// scenarios.py checks both against the expected outputs it computed from each scenario's intent.

#include "gen/ntt.hpp"
#include "include/bundle_tb.h"
#include "include/uint32_array_utils_tb.h"

#include <cstdio>
#include <fstream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: ntt_tb <data_dir> <stage> [scenario]\n");
        return 2;
    }
    const std::string root = argv[1];
    const std::string stage = argv[2];
    const std::string only = (argc > 3) ? argv[3] : "";
    constexpr int W = 32;

    std::ifstream list(root + "/scenarios.txt");
    std::string name;
    int run = 0;
    while (list >> name) {
        if (!only.empty() && name != only) continue;
        const std::string dir = root + "/" + name;

        // 1. What a previous run might have left in the registers.
        ap_uint<1> halted = 1;
        ap_uint<8> error = static_cast<unsigned int>(NttError::BAD_COEFF);
        ap_uint<16> tx_id = 0xFFFF;

        // 2. One activation, on fresh streams.
        hls::stream<streamutils::axi4s_word<W>> s_in, m_out;
        wf::play_stream<W>(dir + "/in", s_in);
        ntt(s_in, m_out, halted, error, tx_id);

        // 3. The response and the status.
        wf::record_stream<W>(m_out, dir + "/" + stage);

        // 4. The host resets the stream path: whatever the kernel left unread is dropped.
        int unread = 0;
        while (!s_in.empty()) { s_in.read(); ++unread; }

        std::ofstream st(dir + "/" + stage + "/status.json");
        st << "{\"halted\": " << (int)halted << ", \"error\": " << (int)error
           << ", \"tx_id\": " << (int)tx_id << ", \"unread_words\": " << unread << "}\n";

        std::printf("scenario %-28s halted=%d error=%d tx_id=%d unread_words=%d\n",
                    name.c_str(), (int)halted, (int)error, (int)tx_id, unread);
        ++run;
    }
    if (run == 0) {
        std::fprintf(stderr, "no scenario ran (data_dir=%s, only=%s)\n", root.c_str(), only.c_str());
        return 1;
    }
    return 0;
}
