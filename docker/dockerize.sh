#!/bin/bash

cd `dirname ${BASH_SOURCE[0]}`

# 和 2025-08-23 的 Colab 保持一致
cuda=12.6.3
pytorch=2.8.0+cu`echo $cuda | awk -F'.' '{print $1 $2}'`
ubuntu=22.04
python=3.12

build_arg_cuda="--build-arg CUDA_VERSION=$cuda"
build_arg_python="--build-arg PYTHON_VERSION=$python"
build_arg_pytorch="--build-arg PYTORCH_VERSION=$pytorch"
build_arg_rust_toolchain="--build-arg RUST_TOOLCHAIN=1.89.0"
build_arg_ubuntu="--build-arg UBUNTU_VERSION=$ubuntu"

build_arg_opts=$build_arg_cuda
build_arg_opts="$build_arg_opts $build_arg_python"
build_arg_opts="$build_arg_opts $build_arg_pytorch"
build_arg_opts="$build_arg_opts $build_arg_rust_toolchain"
build_arg_opts="$build_arg_opts $build_arg_ubuntu"

tag=colab20250823

echo $pytorch | awk -F'+' '{print $2}'

docker build $build_arg_opts -t sammyne/build-a-llm-from-scratch-rs:`git rev-parse --short HEAD`-$tag .
