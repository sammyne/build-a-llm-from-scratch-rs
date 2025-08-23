#!/bin/bash

workdir=/github.com/sammyne/build-a-llm-from-scratch-rs

cargo_home_opt="-v $PWD/_cargo/git:/root/.cargo/git -v $PWD/_cargo/registry:/root/.cargo/registry"

env_libtorch_use_pytorch="-e LIBTORCH_USE_PYTORCH=1"

docker run -it --rm $cargo_home_opt -v $PWD:$workdir -w $workdir sammyne/build-a-llm-from-scratch-rs:b72be49-colab20250823 bash
