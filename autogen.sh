#!/bin/bash
# Bootstrap Autotools for ki7mt-ai-lab-core
set -e

mkdir -p build-aux
autoreconf --install --force
echo ""
echo "Now run: ./configure && make"
