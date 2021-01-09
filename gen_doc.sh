#!/bin/bash

rm -r docs
mkdir docs && cd docs
sphinx-quickstart
# enter
# FaceswapLive
# Karolina Antonik
# 1
# enter
cp ../conf.py.template ./conf.py
sphinx-apidoc -o source/ ../
make html
