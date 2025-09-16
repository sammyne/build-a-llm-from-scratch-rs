ARG GOOGLE_COLAB_VERSION=20250803-ubuntu22.04

# Dockerfile 参见 https://github.com/sammyne/ghcr.io/tree/main/google-colab/20250803/ubuntu22.04
FROM sammyne/google-colab:${GOOGLE_COLAB_VERSION}

ENV RUSTUP_DIST_SERVER="https://rsproxy.cn"     \
  RUSTUP_UPDATE_ROOT="https://rsproxy.cn/rustup"

ARG RUST_TOOLCHAIN=1.89.0

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- --default-toolchain $RUST_TOOLCHAIN -y
