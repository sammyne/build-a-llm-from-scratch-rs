#!/bin/bash

cd `dirname ${BASH_SOURCE[0]}`

googleclab_rev=20250803

build_arg_google_colab="--build-arg GOOGLE_COLAB_VERSION=20250803-ubuntu22.04"
build_arg_rust_toolchain="--build-arg RUST_TOOLCHAIN=1.89.0"

build_arg_opts=$build_arg_google_colab
build_arg_opts="$build_arg_opts $build_arg_rust_toolchain"

docker build $build_arg_opts -t sammyne/build-a-llm-from-scratch-rs:`git rev-parse --short HEAD` .
