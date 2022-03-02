import numpy as np
import pandas as pd
import SimpleITK as sitk
import os
from pathlib import Path
import matplotlib.pyplot as plt

dirpath = r'C:\Users\handsome\Documents\data\20220124\2'
savepath = r'C:\Users\handsome\Documents\data\20220124\max_roi'


def show_img_label(img_array, roi_array, show_index, title):
    show_img = img_array[show_index, ...]
    show_roi = roi_array[show_index, ...]

    plt.title(title)
    plt.imshow(show_img, cmap='gray')
    plt.contour(show_roi)
    #plt.show()



def check_img_label(dir_path):
    case_list = [i for i in  Path(dir_path).iterdir()]
    for i in case_list:
        candidate_roi = [i for i in i.glob('*label.nii')]
        assert len(case_list) != 1, 'Mismatch  label'

        roi_img = sitk.ReadImage(str(candidate_roi[0]))
        roi_array = sitk.GetArrayFromImage(roi_img) # [slice index, x ,y]


        roi_max_index = np.argmax(np.sum(roi_array, axis=(1,2)))
        #roi_index = np.

        candidate_img = [i for i in i.glob('*BSpline*')]
        ori_img_path = str(candidate_roi[0]).replace('-label', '')
        ori_img = sitk.ReadImage(ori_img_path)
        ori_img_array = sitk.GetArrayFromImage(ori_img)
        plt.figure(i.name, figsize=(18, 9))
        plt.subplot(2, 4, 1)
        title = candidate_roi[0].name.replace('-label', '')
        show_img_label(ori_img_array, roi_array, roi_max_index, title)



        k = 1
        for j in candidate_img:
            #plt.figure()
            k = k + 1
            if k > 8:
                plt.subplot(3, 4, k)
            else:
                plt.subplot(2, 4, k)
            sub_img = sitk.ReadImage(str(j))
            img_array = sitk.GetArrayFromImage(sub_img)  # [slice index, x ,y]
            title = j.name[12:-7]
            show_img_label(img_array, roi_array, roi_max_index, title)
        plt.savefig(savepath + '\\' + i.name + '.jpg')
        plt.clf()




def look_direction(dir_path):
    dirs = os.listdir(dir_path)
    for dir in dirs:
        flag = True
        filespath = os.path.join(dir_path, dir)
        for file in os.listdir(filespath):
            if 'label.nii' in file:
                flag = False
                file_path = os.path.join(filespath, file)
                label_direction = get_direction(file_path)
            elif 'BSpline' in file:
                file_path = os.path.join(filespath, file)
                img_direction = get_direction(file_path)
                if img_direction != label_direction:
                    print(dir, ' wrong dirction')
        if flag:
            print(dir, 'wrong')


def get_direction(filepath):
    img = sitk.ReadImage(filepath)
    direction = list(img.GetDirection())
    direction_round = [round(i) for i in direction]
    return direction_round

check_img_label(dirpath)


