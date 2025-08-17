# Build a LLM from scratch (Rust 语言版)

此项目用 [tracel-ai/burn](https://github.com/tracel-ai/burn) 库复现《Build a Large Language Model (From Scratch)》一书的
PyTorch 代码。

## 环境
两种开发模式
1. Debian 12
2. 用 docker/Dockerfile 描述的 docker 容器编译好程序，传到 Google Colab 运行

## 快速开始

每章的代码实现组织在 `crates/chapter{章节序号}` 描述的 rust crate 下（例如，第 2 章的代码在 `crates/chapter02` crate）。

每章的 crate 参照原书的组织风格，将 PyTorch 的代码片段组织为独立的可执行文件，并使用以下命名格式之一
- `030601`：表示第 3 章第 6 小节第 1 子节**仅有**的代码片段；
- `03060202`：表示第 3 章第 6 小节第 2 子节的第 2 个代码片段；

> 目前代码的组织结构还在调整中，并不是每章都满足这个说明。已调整完毕有第 2、3、5、6、7 章。

## 进度
- [x] 01. 理解大型语言模型（Understanding Large Language Models）
- [x] 02. 处理文本数据（Working with Text Data）​
- [x] 03. 编码注意力机制（Coding Attention Mechanisms）​
- [x] 04. 从零实现GPT模型（Implementing a GPT Model from Scratch）​
- [x] 05. 在无标注数据上进行预训练（Pretraining on Unlabeled Data）​​
- [x] 06. 进行文本分类的微调（Fine-tuning for Classification）​
- [x] 07. 进行遵循指令的微调（Fine-tuning to Follow Instructions）​

## 温馨提示
- PyTorch 使用 macOS 的 MPS 会导致计算出错，使得训练误差不会按预期收敛
- burn 的 Module::to_device 转移后的模型不再支持反向传播
- google colab 需要使用下述 python 代码更新 `LD_LIBRARY_PATH` 环境变量，使得编译出来的可执行文件可以找到 `libtorch_cuda.so`、
  `libtorch_cpu.so` 和 `libc10.so` 等库
  ```python
  import os
  import os.path
  import torch

  # 将 pytorch 内置的 libtorch 加到 LD_LIBRARY_PATH
  os.environ['LD_LIBRARY_PATH'] = os.path.dirname(torch.__file__)+'/lib:'+os.environ['LD_LIBRARY_PATH']
  ```
  以下代码无法更新 `LD_LIBRARY_PATH`
  ```
  !export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/torch/lib/:$LD_LIBRARY_PATH
  ```
- 可以白嫖 GPU 资源的地方
  - [Google Colab](https://colab.research.google.com/)
  - https://www.kaggle.com/：如果选用 GPU T4 的话，可以用两块 GPU T4 

## 参考文献
- https://github.com/rasbt/LLMs-from-scratch
