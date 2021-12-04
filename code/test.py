import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
cuda = torch.cuda.is_available()
available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
print(available_gpus )