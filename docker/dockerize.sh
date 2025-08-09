#!/bin/bash

cd `dirname ${BASH_SOURCE[0]}`

build_arg_rust_toolchain="--build-arg RUST_TOOLCHAIN=1.89.0"

docker build $build_arg_rust_toolchaino -t sammyne/build-a-llm-from-scratch-rs:alpha .
