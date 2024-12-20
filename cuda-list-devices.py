#!/usr/bin/env python

import torch

if not torch.cuda.is_available():
    print("No GPUs detected.")

ngpus = torch.cuda.device_count()
for k in range(ngpus):
    print(f"cuda:{k}: {torch.cuda.get_device_name(k)}")
    print(f"    {torch.cuda.get_device_properties(k)}")
