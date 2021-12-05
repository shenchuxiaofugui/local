from torchvision import datasets, transforms
from base import BaseDataLoader
import os
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset



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
    def __init__(self, filepath, modals, use_roi=True, transform=None):
        self.fin_data = []
        self.label = []
        self.roi_data = []
        self.transform=None
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
                        self.roi_data.append(roi_tensor)
                        if use_roi:
                            fin_tensor = data1_tensor * roi_tensor

                        else:
                            fin_tensor = data1_tensor
                        # if transform is not None:
                        #     print(fin_tensor.shape)
                        #     img2 = fin_tensor.numpy().transpose(1, 2, 0)
                        #     img3 = Image.fromarray(np.uint8(img2))
                        #     fin_tensor = transform(img3)
                        fin_tensor = fin_tensor.resize_(fin_tensor.shape[0], 224, 224)
                        self.fin_data.append(fin_tensor)
                        self.label.append(eval(files))

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

    def length(self):
        return len(self.label)

    def select_roi(self, number = 1):
        fin_data = []
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
            fin_data.append(max_slice)
        self.fin_data = fin_data


class ClassDataLoader(BaseDataLoader):

    def __init__(self, data_dir, modals, batch_size, shuffle=True, validation_split=0.0, num_workers=1, use_roi=True, transform=None):
        self.data_dir = data_dir
        self.dataset = CreateNiiDataset(self.data_dir, modals, use_roi, transform)
        self.dataset.select_roi(3)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
