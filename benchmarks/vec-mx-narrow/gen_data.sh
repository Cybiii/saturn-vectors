#!/usr/bin/env bash

g++ gen_data.cpp -I ../common/LoFloat/src -I ../vec-mx-fma -std=c++20 -Wno-overflow -o gen_data 
./gen_data > data.S
rm gen_data