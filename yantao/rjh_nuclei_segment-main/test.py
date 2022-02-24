import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = 'MIG-GPU-4ef25685-6248-e9a8-f4bd-86cbf2763f38/1/0'
import sys
sys.path.append('/homes/ydwang/projects')
from Dataset_all_roi import make_test_loader
import torch
from collections import OrderedDict
import SimpleITK as sitk
from models.network import UNet3D
import matplotlib.pyplot as plt


def load_model(model_path, model, is_multi_GPU=False):
    dict = torch.load(model_path)
    network = dict['model']
    if is_multi_GPU:
        new_dict = OrderedDict()
        for k, v in network.items():
            name = k[7:]
            new_dict[name] = v
        model.load_state_dict(new_dict)
    else:
        model.load_state_dict(network)
    return model


def save_sample(seg_volume, store_path):
    seg_volume = torch.nn.functional.softmax(seg_volume, dim=1)
    seg_volume = seg_volume.cpu().numpy()
    seg_result = np.squeeze(seg_volume)
    seg_result = np.argmax(seg_result, axis=0)
    image = sitk.GetImageFromArray(seg_result)
    sitk.WriteImage(image, store_path)
    return seg_result


def test(params):
    model = UNet3D(input_shape=params['input_shape'], in_channels=1, out_channels=2, init_channels=8)
    network = load_model(params['model_path'], model)
    network = network.cuda()
    network.eval()

    test_loader = make_test_loader(params['data_root'], params['index_path'], params['data_modes'],params['roi_modes'],
                                   params['input_shape'],   is_resize=False, batch_size=params['batch_size'])

    with torch.no_grad():
        for batch_x, batch_y, info in test_loader:
            name = info['case_path'][0].split('/')[-1]
            print(name)
            os.makedirs(os.path.join(params['save_folder'], name), exist_ok=True)
            store_path = os.path.join(params['save_folder'], name,  'seg.nii.gz')
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            output = network(batch_x)
            save_sample(output, store_path)


if __name__ == '__main__':
    params = {
         'data_root': r'/homes/ydwang/Data/seg_result_0104/processs_nii/Group2',
         'index_path': r'/homes/ydwang/Data/seg_result_0104/index/Group2.csv',
         'data_modes': ['qsm.nii'],
         'roi_modes': ['qsm.nii'],
         'input_shape': [256, 256, 136],
         'batch_size': 1,
         'save_folder': r'/homes/ydwang/Data/seg_result_0104/seg_prob/Group2',
         'model_path': r'/homes/ydwang/projects/RJH_Nucleus_seg/Unet_ALL_ROI_logs/model/Epoch119_best_model.pth'
    }

    test(params)

