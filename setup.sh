#!/bin/bash

set -e

cd `dirname ${BASH_SOURCE[0]}`

# 参考 https://pytorch.org/get-started/locally/
LIBTORCH_VERSION=2.7.0

if [[ ! -d third-party/libtorch ]]; then
    mkdir -p third-party
    cd third-party
    if [[ ! -f libtorch-macos-arm64-${LIBTORCH_VERSION}.zip ]]; then
        curl -LO https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-${LIBTORCH_VERSION}.zip
    fi
    unzip libtorch-macos-arm64-${LIBTORCH_VERSION}.zip
    cd -
fi

if [[ ! -f .cargo/config.toml ]]; then
    mkdir -p .cargo
    cat >> .cargo/config.toml <<EOF
[env]
LIBTORCH = ""
EOF
fi

sed -i '' "s|LIBTORCH = .*|LIBTORCH = \"$PWD/third-party/libtorch\"|g" .cargo/config.toml
