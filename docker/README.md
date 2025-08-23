# docker

快速查看 Colab 的工具版本的 Jupyter Notebook 如下

```
!cat /etc/os-release

!python --version
```

```
from importlib.metadata import version

pkgs = ["matplotlib", 
        "numpy", 
        "tiktoken", 
        "torch",
        "tensorflow" # For OpenAI's pretrained weights
       ]
for p in pkgs:
    print(f"{p} version: {version(p)}")
```
