import json
import numpy as np

from gpt_download import download_and_load_gpt2

settings, params = download_and_load_gpt2(model_size="355M", models_dir="gpt2")

print("Settings:", settings)
print("Parameter dictionary keys:", params.keys())

print(params["wte"])
print("Token embedding weight tensor dimensions:", params["wte"].shape)

path = "gpt2/355M/params-355m.json"
with open(path, "w", encoding="utf-8") as f:
    # json.dump(params, f, ensure_ascii=False)
    json.dump(params, f, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
print(f"Parameters saved to {path}")