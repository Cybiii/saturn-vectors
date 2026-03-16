#!/usr/bin/env bash
# Generate Spike MX golden for vec-mx-axpy. Requires spike with MX extensions.
ISA=rv64imafdcv_zfh_zvfh
SPIKE_ISA=${ISA}_zvfbfa_zfbfmin_zvfofp8min

riscv64-unknown-elf-gcc gen_data.c ../../common-data-gen/mx_data_gen.c -I ../../common -I ../../common-data-gen -march=$ISA -o gen_data
spike --isa=$SPIKE_ISA ${LOG:+-l --log=$LOG} pk gen_data > ../data.S.tmp && mv ../data.S.tmp ../data.S
rm -f gen_data
