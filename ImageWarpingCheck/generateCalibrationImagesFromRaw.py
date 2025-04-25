import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk


fFiles = [r"", #PATH TO DOTS *_images.dat; get these files from calibaration during installation
          r"", #PATH TO CIRCLES *_images.dat; get these files from calibaration during installation
          r""  #PATH TO SQUARES *_image.dat; get these files from calibaration during installation
]

names = ["Dots", "Circles", "Squares"]


def processFile(fFile, name):

    with open(fFile, "rb") as f:
        dat = np.fromfile(f, np.int8)

    # Skip first 400 header bytes

    dat = dat[400:]

    numImg = int(len(dat) / (2448 * 2048))
    ddat = dat.reshape((numImg, 2048, 2448))

    # extract every even (filtered image)
    evenImgNrs = np.array([i for i in range(1, numImg, 2)])

    filteredImgs = np.mean(ddat[evenImgNrs, ::], axis=0)

    sitk.WriteImage(sitk.GetImageFromArray(filteredImgs), f"{name}.nii")


if __name__ == '__main__':

    assert( len(fFiles) == len(names) )

    for i in range(len(fFiles)):
        processFile(fFiles[i], names[i])
