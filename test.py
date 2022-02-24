import pandas as pd
import os
from adapthist import equalize_adapthist
import SimpleITK as sitk
import numpy as np
from skimage import img_as_float
filepath = r'\\mega\syli\python\nnunet\nnUNet_raw_data\nnUNet_raw_data\Task002_test\imagesTs'
for file in os.listdir(filepath):
    img = sitk.ReadImage(os.path.join(filepath, file))
    img_arr = sitk.GetArrayFromImage(img)
    img_arr = img_arr[img_arr>0]
    #img_arr = (img_arr - np.mean(img_arr))/np.std(img_arr)
    print(file, np.mean(img_arr))




