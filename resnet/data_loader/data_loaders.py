from torchvision import datasets, transforms
from base import BaseDataLoader
import os
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import numpy as np
import logging
import torchvision.transforms.functional as F


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class CreateNiiDataset(Dataset):
    def __init__(self, filepath, modals, use_roi=True):
        self.fin_data = []
        self.label = []
        self.roi_data = []
        self.transform = None
        self.casename = []
        for files in os.listdir(filepath):
            paths = os.path.join(filepath,files)
            lines = os.listdir(paths)  #os.path.join(self.path1, self.A)
            for line in lines:
                imgs = os.listdir(os.path.join(paths, line))
                for single_img in imgs:
                    if single_img[:-4] in modals:
                        img1 = sitk.ReadImage(os.path.join(paths, line, single_img))
                        data1 = sitk.GetArrayFromImage(img1)
                        data1_tensor = torch.from_numpy(data1)
                        roi_img = sitk.ReadImage(os.path.join(paths, line, single_img[:-4]+'_roi.nii.gz'))
                        roi_data = sitk.GetArrayFromImage(roi_img)
                        roi_tensor = torch.from_numpy(roi_data/1.0)
                        if use_roi:
                            fin_tensor = data1_tensor * roi_tensor

                        else:
                            fin_tensor = data1_tensor
                        fin_tensor = F.resize(fin_tensor, [224, 224], interpolation=3)
                        self.roi_data.append(roi_tensor)
                        self.fin_data.append(fin_tensor)
                        self.label.append(eval(files))
                        self.casename.append(files+single_img[:-4])

    def __getitem__(self, index):
        data = self.fin_data[index]
        label = torch.LongTensor([self.label[index]])
        label = torch.squeeze(label)
        if torch.cuda.is_available():
            data = data.cuda()
            label = label.cuda()

        return data, label

    def __len__(self):
        return len(self.label)


    def transformer(self, transforms):
        for transform in transforms:
            fin_tensor = transform(self.fin_data)
            roi_tensor = transform(self.roi_data)
            self.fin_data.append(fin_tensor)
            self.roi_data.append(roi_tensor)

    def select_roi(self, number = 1):
        fin_data = []
        fin_roi = []
        i = 0
        for roi_img, data_img in zip(self.roi_data, self.fin_data):
            roi_size = []
            for slice in range(roi_img.shape[0]):
                roi_size.append(torch.sum(roi_img[slice, ...]))
            a = roi_size.index(max(roi_size)) - (number - 1) / 2
            b = a + number
            if b >= roi_img.shape[0]:
                temp = b - roi_img.shape[0]
                a = a - temp - 1
                b = b - temp - 1
            c = range(int(a), int(b))
            indices = torch.LongTensor(c)
            # print(c)
            max_slice = torch.index_select(data_img, 0, indices).type(torch.cuda.FloatTensor)
            max_roi = torch.index_select(roi_img, 0, indices).type(torch.cuda.FloatTensor)
            fin_data.append(max_slice)
            fin_roi.append(max_roi)
        self.fin_data = fin_data
        self.roi_data = fin_roi

    def daochu(self, path):
        for i in range(len(self.label)):
            img = self.fin_data[i].cpu()
            img = np.uint8(img).transpose(1, 2, 0)
            img = Image.fromarray(img)
            print(path + f'\\imgs\\{self.casename[i]}.jpg')
            print(path + f'/imgs/{self.casename[i]}.jpg')
            img.save(path + f'/imgs/{self.casename[i]}.jpg')
            roi = self.roi_data[i].cpu()
            roi = np.uint8(roi).transpose(1, 2, 0)
            roi = Image.fromarray(roi)
            roi.save(path + f'/masks/{self.casename[i]}.jpg')


class ClassDataLoader(BaseDataLoader):
    def __init__(self, data_dir, modals, batch_size, shuffle=True, validation_split=0.0, num_workers=1, use_roi=True):
        self.data_dir = data_dir
        self.dataset = CreateNiiDataset(self.data_dir, modals, use_roi)
        self.dataset.select_roi(3)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [os.path.splitext(file)[0] for file in os.listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        if img_ndarray.ndim == 2 and not is_mask:
            img_ndarray = img_ndarray[np.newaxis, ...]
        elif not is_mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        if not is_mask:
            img_ndarray = img_ndarray / 255

        return img_ndarray

    @classmethod
    def load(cls, filename):
        ext = os.path.splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])

        assert img.size == mask.size, \
            'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class UnetDataLoader(BaseDataLoader):

    def __init__(self, img_dir, roi_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, use_roi=True, transform=None):
        self.dataset = BasicDataset(img_dir, roi_dir, mask_suffix='_roi')
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
