import os
import SimpleITK as sitk
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image


class CreateNiiDataset(Dataset):
    def __init__(self, filepath, modals, use_roi=True, transform=None):
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
                        if transform is not None:
                            # img2 = fin_tensor.numpy().transpose(1, 2, 0)
                            # img3 = Image.fromarray(np.uint8(img2))
                            fin_tensor = transform(fin_tensor)
                            roi_tensor = transform(roi_tensor)
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

