import pandas as pd
from adapthist import equalize_adapthist
import SimpleITK as sitk
import numpy as np
from skimage import img_as_float
# a = pd.read_csv(r'\\mega\syli\dataset\Primary and metastatic\test\T1CE\T1CE_feature_2.csv')
# b = pd.read_csv(r'\\mega\syli\dataset\Primary and metastatic\test\T1CE\T1CE_feature_4.csv')
# #print(a['CaseName'].sum())
# for index in a.columns:
#     if a[index].sum() == b[index].sum():
#         print(index)
# img=sitk.ReadImage(r'\\mega\syli\dataset\Primary and metastatic\Primary\102731520\T1.nii')
# ori_arr=sitk.GetArrayFromImage(img)
# ori_arr = img_as_float(ori_arr)
# print(ori_arr.min(), ori_arr.max())
# np.place(ori_arr, ori_arr == 0, np.nan)
# ori_arr = equalize_adapthist(ori_arr)
# print(ori_arr)

# minmax_arr = (ori_arr - np.min(ori_arr)) / (np.max(ori_arr) - np.min(ori_arr))
# equalize_adapthist(minmax_arr)
#print(np.iinfo('uint16').max)




