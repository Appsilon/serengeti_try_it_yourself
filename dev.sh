#!/usr/bin/env bash

#1. Bundle

pyinstaller main.py --clean --noconfirm \
     --additional-hooks-dir=hooks

#2. Go into dist dir
pushd dist/main


#3. Run
LD_LIBRARY_PATH=torch/lib/ ./main ../../images_fun_examples

popd
