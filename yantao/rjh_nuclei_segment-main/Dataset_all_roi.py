import sys
sys.path.append('/homes/ydwang/projects')
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import volumentations as volumentations
import pandas as pd
from utils.my_utils import load_data, normalize
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
from MeDIT.SaveAndLoad import LoadNiiData


class Datasets3D(Dataset):
    def __init__(self, data_folder, csv_file, data_modes, roi_modes, input_shape,
                 is_training=True, is_normalize=True, is_resize=True):
        self.data_folder = data_folder
        self._modalities = data_modes
        self.roi_modes = roi_modes
        self.resize_shape = input_shape
        self.is_training = is_training
        self.is_normlize = is_normalize
        self.is_resize = is_resize
        self.csv_file = csv_file
        self.case_list = self.get_case_list(self.csv_file)

    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, index):
        case = self.case_list[index]
        case = case.replace('\t', '')
        path = os.path.join(self.data_folder, case)
        data_list, seg_list = self.get_data(path)
        data_volumes = np.stack(data_list, axis=-1).astype('float32')
        seg_volumes = np.stack(seg_list, axis=-1).astype("float32")
        if self.is_resize:
            aug_data_volumes, aug_seg_volumes = self.aug_sample(data_volumes, seg_volumes, is_training=self.is_training)
        else:
            aug_data_volumes = data_volumes
            aug_seg_volumes = seg_volumes

        normlized_data_volumes = self.normlize_data(aug_data_volumes, is_normlize=self.is_normlize)
        # data_input, mask_input: [c, h, w, d]
        data_input = np.transpose(normlized_data_volumes, [3, 0, 1, 2]).astype('float32')
        mask_input = np.transpose(aug_seg_volumes, [3, 0, 1, 2]).astype('float32')

        image_shape = np.shape(data_list[0])

        case_info = {"case_path":path, "original_shape": image_shape}
        return (torch.tensor(data_input.copy(), dtype=torch.float),
                torch.tensor(mask_input.copy(), dtype=torch.float),
                case_info)

    def aug_sample(self, image, mask, is_training=True):
        """
        vol: [H, W, D(, C)]

        x, y, z <--> H, W, D

        you should give (H, W, D) form shape.

        skimage interpolation notations:

        order = 0: Nearest-Neighbor
        order = 1: Bi-Linear (default)
        order = 2: Bi-Quadratic
        order = 3: Bi-Cubic
        order = 4: Bi-Quartic
        order = 5: Bi-Quintic

        Interpolation behaves strangely when input of type int.
        ** Be sure to change volume and mask data type to float !!! **
        I change resize in functionals.py!!!
        """
        image = np.float32(image)
        mask = np.float32(mask)
        if is_training:
            train_tranform = volumentations.Compose([
                volumentations.Resize(self.resize_shape, interpolation=1, always_apply=True, p=1.0),
                # volumentations.RandomCrop(self.resize_shape, p=1),
                volumentations.Rotate((-5, 5), (-5, 5), (0, 0), interpolation=1, p=0.2),
                volumentations.Flip(0, p=0.5),
                volumentations.Flip(1, p=0.5),
                volumentations.Flip(2, p=0.5),
                # volumentations.RandomScale(scale_limit=[0.9, 1.1], interpolation=1, p=0.5),
                # volumentations.PadIfNeeded(self.resize_shape, border_mode='constant', always_apply=True, p=1.0),
                volumentations.ElasticTransform((0, 0.1), interpolation=1, p=0.2),
                volumentations.RandomGamma((0.7, 1.2), p=0.2),
            ], p=1.0)
        else:
            train_tranform = volumentations.Compose([
                volumentations.Resize(self.resize_shape, interpolation=1, always_apply=True, p=1.0)], p=1.0)
        data = {'image': image, 'mask': mask}
        transformed_data = train_tranform(**data)
        aug_image = transformed_data['image']
        aug_mask = transformed_data['mask']
        return aug_image, aug_mask

    def get_case_list(self, csv_file):
        df = pd.read_csv(csv_file, encoding='gbk',dtype='object')
        id = df['ID'].values
        case_list = list(id)
        return case_list

    def get_data(self, path):
        # the shape of data_list: [[H, W, D], [H, W, D], ......]
        # the shape of roi_list: [[H, W, D]]
        files = os.listdir(path)
        data_list = []
        roi_list = []
        for modality in self._modalities:
            data_file = [x for x in files if modality in x][0]
            data_path = os.path.join(path, data_file)
            data = load_data(data_path)
            data_list.append(data)

        for roi_mode in self.roi_modes:
            roi_file = [x for x in files if roi_mode in x][0]
            roi_path = os.path.join(path, roi_file)
            roi_data = load_data(roi_path)
            roi_list.append(roi_data)
        return data_list, roi_list

    def normlize_data(self, data, is_normlize=True):
        if is_normlize:
            num = np.shape(data)[-1]
            n_data_list = [normalize(data[..., i]) for i in range(num)]
            n_data = np.stack(n_data_list, axis=3)
        else:
            n_data = data
        return n_data


def make_data_loaders(data_root, train_csv_file, val_csv_file, data_modes, roi_modes, input_shape, batch_size=8):
    train_ds = Datasets3D(data_root, train_csv_file, data_modes, roi_modes, input_shape)
    val_ds = Datasets3D(data_root, val_csv_file, data_modes, roi_modes, input_shape, is_training=False)
    loaders = {}
    loaders['train'] = DataLoader(train_ds, batch_size=batch_size,
                              num_workers=0, shuffle=True, drop_last=True)
    loaders['eval'] = DataLoader(val_ds, batch_size=batch_size,
                             num_workers=0, shuffle=True, drop_last=True)
    return loaders


def make_test_loader(data_root, test_csv_path, data_modes, roi_modes, input_shape,  is_resize=True, batch_size=1):
    test_ds = Datasets3D(data_root, test_csv_path, data_modes, roi_modes, input_shape, is_training=False, is_resize=is_resize)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                                 num_workers=0, shuffle=False)
    return test_loader


if __name__ == "__main__":
    data_root =  r'/homes/syli/dataset/EC_seg/EC-old1'
    index_path =r'/homes/syli/python/yantao/dada/index'
    data_modes =  ['T1CE.nii']
    roi_modes =  ['T1CE_roi.nii.gz']
    input_shape = [320, 320, 32]
    train_csv_path = os.path.join(index_path, 'train_index.csv')
    val_csv_path = os.path.join(index_path, 'val_index.csv')
    test_csv_path = os.path.join(index_path, 'test_index.csv')
    # loaders = make_data_loaders(data_root, train_csv_path, val_csv_path, data_modes, roi_modes, input_shape, batch_size=1)
    test_loader = make_test_loader(data_root, train_csv_path, data_modes, roi_modes, input_shape, is_resize=False, batch_size=1)
    for x, y, z in test_loader:
        images_data = x.numpy()
        seg_data = y.numpy()
        print(images_data.shape, seg_data.shape)
        print(np.max(seg_data))
        print(np.mean(images_data[0, 0, ...]))
        import matplotlib.pyplot as plt
        plt.imshow(images_data[0, 0, 10, :, :], cmap='gray')
        #plt.show()
        plt.imshow(seg_data[0, 0, 15,:, :], cmap='gray')
        plt.show()




