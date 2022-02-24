from MeDIT.SaveAndLoad import LoadNiiData
from MeDIT.Visualization import Imshow3DArray
from numpy import fft


def image2kspace(data):
    k_data = fft.fftshift(fft.fftn(fft.ifftshift(data)))
    return k_data

def kspace2image(k_data):
    image = fft.fftshift(fft.ifftn(fft.ifftshift(k_data)))
    return image.real


def truncted_in_kspace(data, z_range=(8, -8)):
    k_data = image2kspace(data)
    truncted_k_data = k_data[:, :, z_range[0]:z_range[1]]
    truncted_data = kspace2image(truncted_k_data)
    return truncted_data

