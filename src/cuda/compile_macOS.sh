#/bin/bash

srcfile=$1
modname=`echo $1 | cut -d. -f1`

c++ -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup `python3 -m pybind11 --includes` $srcfile -o $modname`python3-config --extension-suffix`
