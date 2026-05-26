import os
import torch

print("inside CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)
print("torch cuda count:", torch.cuda.device_count(), flush=True)

for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i), flush=True)
