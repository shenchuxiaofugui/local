import SimpleITK as sitk
import numpy as np


def resampleSpacing(sitkImage, newspace=(1,1,1)):
    euler3d = sitk.Euler3DTransform()
    xsize, ysize, zsize = sitkImage.GetSize()
    xspacing, yspacing, zspacing = sitkImage.GetSpacing()
    origin = sitkImage.GetOrigin()
    direction = sitkImage.GetDirection()
    #新的X轴的Size = 旧X轴的Size *（原X轴的Spacing / 新设定的Spacing）
    new_size = (int(xsize*xspacing/newspace[0]),int(ysize*yspacing/newspace[1]),int(zsize*zspacing/newspace[2]))
    #如果是对标签进行重采样，模式使用最近邻插值，避免增加不必要的像素值
    sitkImage = sitk.Resample(sitkImage,new_size,euler3d,sitk.sitkNearestNeighbor,origin,newspace,direction) #sitk.sitkNearestNeighbor
    return sitkImage

#读取nifit原数据 ，size为：(880, 880, 12)
NifitmPath = r'\\mega\syli\python\jiaoben\test_code\test\0\5341777\T1_roi.nii'
sitkImage = sitk.ReadImage(NifitmPath)
print("重采样前的信息")
print("尺寸：{}".format(sitkImage.GetSize()))
print("体素大小(x,y,z):{}".format(sitkImage.GetSpacing()))

print('='*30+'我是分割线'+'='*30)

newResample = resampleSpacing(sitkImage, newspace=[1,1,6])
sitk.WriteImage(newResample, r'\\mega\syli\python\jiaoben\test_code\hahah_roi.nii')
print("重采样后的信息")
print("尺寸：{}".format(newResample.GetSize()))
print("体素大小(x,y,z):{}".format(newResample.GetSpacing()) )

"""
统一Size
X轴和Y轴的Size和Spacing没有变化，
Z轴的Size和Spacing有变化
"""
def resampleSize(sitkImage, depth):
    #重采样函数
    euler3d = sitk.Euler3DTransform()

    xsize, ysize, zsize = sitkImage.GetSize()
    xspacing, yspacing, zspacing = sitkImage.GetSpacing()
    new_spacing_z = zspacing/(depth/float(zsize))

    origin = sitkImage.GetOrigin()
    direction = sitkImage.GetDirection()
    #根据新的spacing 计算新的size
    newsize = (xsize,ysize,int(zsize*zspacing/new_spacing_z))
    newspace = (xspacing, yspacing, new_spacing_z)
    sitkImage = sitk.Resample(sitkImage,newsize,euler3d,sitk.sitkNearestNeighbor,origin,newspace,direction)
    return sitkImage

# DEPTH = 16  #需要重采样Size的层数
#
# #读取nifit原数据 ，size为：(880, 880, 12)
# print("重采样前的信息")
# print("尺寸：{}".format(sitkImage.GetSize()))
# print("体素大小(x,y,z):{}".format(sitkImage.GetSpacing()) )
#
# print('='*30+'我是分割线'+'='*30)
#
#
# newsitkImage = resampleSize(sitkImage, depth=DEPTH)
# print("重采样后的信息")
# print("尺寸：{}".format(newsitkImage.GetSize()))
# print("体素大小(x,y,z):{}".format(newsitkImage.GetSpacing()) )

# image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
# image_array = exposure.equalize_adapthist(image_array)