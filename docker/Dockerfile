# 注意 cuda 版本需要和 Google Colab 匹配
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

RUN sed -i 's/archive.ubuntu.com/mirrors.tencent.com/g' /etc/apt/sources.list &&\
  sed -i 's/security.ubuntu.com/mirrors.tencent.com/g' /etc/apt/sources.list

ENV RUSTUP_DIST_SERVER="https://rsproxy.cn"     \
  RUSTUP_UPDATE_ROOT="https://rsproxy.cn/rustup"

RUN apt update                                                                    &&\
  apt install -y python3.11                                                       &&\
  apt install -y curl git pkg-config libfontconfig-dev libssl-dev                                   &&\
  update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 100  &&\
  apt clean                                                                       &&\
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ARG RUST_TOOLCHAIN=1.89.0

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- --default-toolchain $RUST_TOOLCHAIN -y

RUN curl -LO -o get-pip.py https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py && rm get-pip.py

RUN pip config set global.index-url http://mirrors.cloud.tencent.com/pypi/simple &&\
  pip config set global.trusted-host mirrors.cloud.tencent.com

# # 安装 PyTorch GPU 版本（需和 google colab 的 CUDA 版本匹配）
RUN pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124

# 指示 tch-rs 库使用 PyTorch 关联的 libtorch。
# 参考：https://github.com/LaurentMazare/tch-rs?tab=readme-ov-file#python-pytorch-install
ENV LIBTORCH_USE_PYTORCH=1
