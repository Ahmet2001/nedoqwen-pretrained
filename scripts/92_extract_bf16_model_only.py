import sys
from pathlib import Path
import torch

src = Path(sys.argv[1])
dst = Path(sys.argv[2])

print("SRC:", src)
print("DST:", dst)

ckpt = torch.load(src, map_location="cpu", weights_only=False)

if not isinstance(ckpt, dict) or "model" not in ckpt:
    raise ValueError("Expected checkpoint dict with key 'model'")

state = ckpt["model"]
state_bf16 = {}

for k, v in state.items():
    if torch.is_tensor(v):
        state_bf16[k] = v.to(torch.bfloat16)
    else:
        state_bf16[k] = v

out = {
    "model": state_bf16,
    "step": ckpt.get("step", None),
    "base_ckpt": ckpt.get("base_ckpt", None),
    "base_step": ckpt.get("base_step", None),
    "sft_data": ckpt.get("sft_data", None),
    "dtype": "bfloat16",
    "format": "model_only_bf16_state_dict"
}

torch.save(out, dst)

print("WROTE:", dst)
print("SIZE_GB:", round(dst.stat().st_size / 1e9, 4))
