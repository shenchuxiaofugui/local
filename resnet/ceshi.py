from data_loader.data_loaders import ClassDataLoader
from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
a = ClassDataLoader("/homes/syli/dataset/resnet/EC_all", [ "T1CE","DWI"], 100, shuffle=True, validation_split=0.2, num_workers=0, use_roi=False, transform=None)
print('successful')
for batch_idx, (data, label) in tqdm(enumerate(a)):
    print(data, label)

