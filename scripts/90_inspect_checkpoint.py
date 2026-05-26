import sys
import json
from pathlib import Path

import torch

path = Path(sys.argv[1])
print("CHECKPOINT:", path)
print("EXISTS:", path.exists())
print("SIZE_GB:", round(path.stat().st_size / 1e9, 4))

print("\nLoading checkpoint...")
ckpt = torch.load(path, map_location="cpu", weights_only=False)
print("LOADED TYPE:", type(ckpt))

def describe_tensor(name, t):
    print(
        f"{name}: shape={tuple(t.shape)} dtype={t.dtype} "
        f"numel={t.numel()} size_mb={t.numel() * t.element_size() / 1e6:.2f}"
    )

if isinstance(ckpt, dict):
    print("\nTOP-LEVEL KEYS:")
    for k in ckpt.keys():
        v = ckpt[k]
        if torch.is_tensor(v):
            print(f"- {k}: Tensor {tuple(v.shape)} {v.dtype}")
        elif isinstance(v, dict):
            print(f"- {k}: dict with {len(v)} keys")
        else:
            print(f"- {k}: {type(v)} -> {repr(v)[:200]}")

    state = None
    state_key = None
    for candidate in ["model", "model_state_dict", "state_dict", "module", "net"]:
        if candidate in ckpt and isinstance(ckpt[candidate], dict):
            state = ckpt[candidate]
            state_key = candidate
            break

    if state is None:
        tensor_values = [v for v in ckpt.values() if torch.is_tensor(v)]
        if len(tensor_values) > 10:
            state = ckpt
            state_key = "<raw_state_dict>"

    if state is not None:
        print(f"\nSTATE_DICT KEY: {state_key}")
        print("NUM ITEMS:", len(state))

        total_params = 0
        dtype_counts = {}

        print("\nFIRST 50 ITEMS:")
        shown = 0
        for k, v in state.items():
            if torch.is_tensor(v):
                total_params += v.numel()
                dtype_counts[str(v.dtype)] = dtype_counts.get(str(v.dtype), 0) + v.numel()
                if shown < 50:
                    describe_tensor(k, v)
                    shown += 1
            else:
                if shown < 50:
                    print(f"{k}: non-tensor {type(v)} -> {repr(v)[:120]}")
                    shown += 1

        print("\nTOTAL PARAMS:", total_params)
        print("TOTAL PARAMS_B:", round(total_params / 1e9, 6))
        print("DTYPE PARAM COUNTS:", json.dumps(dtype_counts, indent=2, ensure_ascii=False))
    else:
        print("\nCould not identify model state_dict.")
else:
    print("Checkpoint is not a dict.")
