import os
import numpy as np
import SimpleITK as sitk
casepath = r'\\mega\syli\dataset\zj_data\jly\data'
dir1 = os.listdir(casepath)
dir2 = os.listdir(r'C:\Users\handsome\Documents\data\20220124\1')
print(len(set(dir2).difference(set(dir1))))
#a = pd.DataFrame(columns=list(set(dir2).difference(set(dir1))))
#a.to_excel(r'\\mega\syli\dataset\zj_data\queshao.xlsx')






