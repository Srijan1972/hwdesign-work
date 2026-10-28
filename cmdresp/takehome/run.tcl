# run.tcl -- Vitis HLS for the batched NTT-256 accelerator.
#
#   WAVEFLOW_NTT_STAGE=csim    every scenario, C simulation
#   WAVEFLOW_NTT_STAGE=synth   C synthesis, then ONE RTL co-simulation run through every scenario
#
#   vitis-run --mode hls --tcl run.tcl
#
# (The stage is an environment variable because vitis-run has no --tclargs.)
#
# The kernel boundary (gen/ntt.cpp, gen/ntt.hpp) is generated; the kernel body
# (ntt_body_impl.tpp, included from gen/ntt.hpp), the twiddle ROM (ntt_zetas.h, written by
# ntt_build.py) and the testbench (ntt_tb.cpp) are not.  The testbench writes each scenario's
# response to data/<scenario>/<stage>/.  The project, prj_ntt, is one directory deep (Vitis
# drops the kernel from csim when a project is nested deeper) and is not committed.

set script_dir [file dirname [file normalize [info script]]]
set stage ""
if {[info exists ::env(WAVEFLOW_NTT_STAGE)]} {
    set stage $::env(WAVEFLOW_NTT_STAGE)
}
if {$stage ni {csim synth}} {
    puts "WAVEFLOW_ERROR: set WAVEFLOW_NTT_STAGE to csim or synth (got '$stage')."
    exit 1
}
set top ntt

open_project -reset prj_ntt
set_top $top
add_files gen/ntt.cpp -cflags "-I."
add_files -tb ntt_tb.cpp -cflags "-I."
set streamutils_cpp [file join $script_dir "include" "streamutils.cpp"]
if {[file exists $streamutils_cpp]} {
    add_files -tb $streamutils_cpp
}

open_solution -reset "solution1"
set_part {xc7z020clg484-1}
create_clock -period 10
set data_dir [file join $script_dir "data"]

if {$stage eq "csim"} {
    if {[catch {csim_design -argv "$data_dir csim"} res]} {
        puts "WAVEFLOW_ERROR: $top C simulation failed."
        puts $res
        exit 1
    }
    puts "WAVEFLOW_SUCCESS: $top C simulation passed."
    exit 0
}

if {[catch {csynth_design} res]} {
    puts "WAVEFLOW_ERROR: $top C synthesis failed."
    puts $res
    exit 1
}
if {[catch {cosim_design -argv "$data_dir cosim" -trace_level none} res]} {
    puts "WAVEFLOW_ERROR: $top RTL co-simulation failed."
    puts $res
    exit 1
}
puts "WAVEFLOW_SUCCESS: $top C synthesis and RTL co-simulation of every scenario passed."
exit 0
