import json
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import matplotlib.pyplot as plt
import os
from pathlib import Path
import shutil


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    #n_gpu_use = len(gpus)
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device(f'cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


def load_checkpoint_1(model, checkpoint='No', optimizer='', loadOptimizer=False):
    if checkpoint != 'No':
        print("loading checkpoint...")
        model_dict = model.state_dict()
        print(','.join(map(str, sorted(model_dict.keys()))))
        #modelCheckpoint = torch.load(checkpoint)
        pretrained_dict = torch.load(checkpoint)
        #pretrained_dict = modelCheckpoint.state_dict()
        # 过滤操作
        new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
        print(','.join(map(str, sorted(new_dict.keys()))))
        model_dict.update(new_dict)
        # 打印出来，更新了多少的参数
        print('Total : {}, update: {}'.format(len(pretrained_dict), len(new_dict)))
        model.load_state_dict(model_dict)
        print("loaded finished!")
        # 如果不需要更新优化器那么设置为false
    else:
        print('No checkpoint is included')
    return model, optimizer


def metric_visualization(name, epoch, losses, path):
    plt.title(name)
    plt.xlabel("epoch")
    plt.ylabel(name)
    plt.plot(range(epoch + 1), losses)
    plt.legend([name, f"valid_{name}"])
    plt.savefig(path, dpi=300)


def split_dirs(sourcepath, storepath, formpath):
    df = pd.read_excel(formpath).iloc[:, :2]
    df = df.astype(dtype='str')
    if not os.path.exists(os.path.join(storepath, '0')):
        os.makedirs(os.path.join(storepath, '0'))
        os.makedirs(os.path.join(storepath, '1'))
    for index,row in df.iterrows():
        dir = os.path.join(sourcepath, row['ID'])
        storedir = os.path.join(storepath, row['淋巴结转移阳性'], row['ID'])
        shutil.copytree(dir, storedir)

