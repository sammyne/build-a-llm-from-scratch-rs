# 05. Pretraining on unlabeled data

## 温馨提示
- 苹果电脑的 MPS 框架有问题，导致 PyTorch 和 burn 的训练过程的误差无法正确降低
- burn 库的 Module::to_device 转移后的模型不再支持反向传播
