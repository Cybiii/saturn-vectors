#!/usr/bin/env bash

g++ gen_data.cpp -I ../common/LoFloat/src -std=c++20 -Wno-overflow -o gen_data 
./gen_data > data.S
rm gen_data