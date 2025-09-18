#!/bin/bash

workdir=/github.com/sammyne/build-a-llm-from-scratch-rs

cargo_home_opt="-v $PWD/_cargo/git:/root/.cargo/git -v $PWD/_cargo/registry:/root/.cargo/registry"

env_opts="-e LIBTORCH_USE_PYTORCH=1 -e LIBTORCH_BYPASS_VERSION_CHECK=1"

docker run -it --rm $env_opts $cargo_home_opt -v $PWD:$workdir -w $workdir sammyne/build-a-llm-from-scratch-rs:d6097db bash
