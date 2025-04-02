from os.path import join as pjoin

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import fill_voids
from skimage.exposure import equalize_adapthist, rescale_intensity
import matplotlib.ticker as tkr


cnorm = {
    "mu": np.array([8.74108109, -0.12440419, 0.0444982]),
    "sigma": np.array([0.6135447, 0.10989545, 0.0286032]),
}

######NZ: I don't need the following lines for PAS stain ################
stain_unmixing_routine_params = {
    "stains": ["hematoxylin", "eosin"],
    "stain_unmixing_method": "macenko_pca",
}
#########################################################################


W_target = np.array(
    [
        [0.5807549, 0.08314027, 0.08213795],
        [0.71681094, 0.90081588, 0.41999816],
        [0.38588316, 0.42616716, -0.90380025],
    ]
)

######## NZ: No modifications are required for PAS stain ####################################
def getMask(img: sitk.Image) -> sitk.Image:

    imggilt = sitk.SmoothingRecursiveGaussian(img, (15, 15, 1), True)
    npmsk = sitk.GetArrayFromImage(sitk.OtsuThreshold(imggilt, 0, 1)).astype(np.uint8)
    npmsk = fill_voids.fill(npmsk)
    msk = sitk.GetImageFromArray(npmsk)
    msk.CopyInformation(imggilt)

    return msk
############################################################################################


######### NZ: No modifications are required for PAS stain ##################################
def normalizeImageRange(img: sitk.Image, msk: sitk.Image, range=(0.1, 99.9)) -> sitk.Image:

    npImg = sitk.GetArrayFromImage(img)
    npMsk = sitk.GetArrayFromImage(msk)

    npImg = npImg - np.percentile(npImg[npMsk != 0], range[0])
    npImg = npImg / np.percentile(npImg[npMsk != 0], range[1])
    npImg[npImg < 0] = 0

    retImg = sitk.GetImageFromArray(npImg)
    retImg.CopyInformation(img)

    return retImg
#############################################################################################



######### NZ: No modifications are required for PAS stain ##################################
def resizeImg(img, new_size, interpolator=None) -> sitk.Image:

    reference_image = sitk.Image(new_size, img.GetPixelIDValue())
    reference_image.SetOrigin(img.GetOrigin())
    reference_image.SetDirection(img.GetDirection())
    reference_image.SetSpacing([ sz * spc / nsz for nsz, sz, spc in zip(new_size, img.GetSize(), img.GetSpacing())])


    if interpolator is None:
        return sitk.Resample(img, reference_image)
    else:
        return sitk.Resample(img, reference_image, sitk.Transform(), interpolator)
############################################################################################    



############ NZ: Modifications for PAS ###################################################################
def Mic2HE(DAPI: sitk.Image, Phalloidin: sitk.Image) -> sitk.Image:

    msk = getMask(Phalloidin)
    npNuclearStain = sitk.GetArrayFromImage(DAPI).astype(np.float32)
    npCytosolStain = sitk.GetArrayFromImage(Phalloidin).astype(np.float32)  # *1.25

    p2, p98 = np.percentile(npNuclearStain[sitk.GetArrayViewFromImage(msk)!=0], (.005, 99.3))
    npNuclearStain = rescale_intensity(npNuclearStain, in_range=(p2, p98), out_range=(0,1))


    p2, p98 = np.percentile(npCytosolStain[sitk.GetArrayViewFromImage(msk)!=0], (.005, 99.5))
    npCytosolStain = rescale_intensity(npCytosolStain, in_range=(p2, p98), out_range=(0,1))


    gammaNuc = 1
    gammaCyt = 1
    A, B = .7, 1.5

    npNuclearStain = A * npNuclearStain ** (gammaNuc)
    npCytosolStain = B * (npCytosolStain) ** (gammaCyt)

    R = 1 - npNuclearStain * (1 - 0.24) - npCytosolStain * (1 - 0.88)
    G = 1 - npNuclearStain * (1 - 0.21) - npCytosolStain * (1 - 0.27)
    B = 1 - npNuclearStain * (1 - 0.62) - npCytosolStain * (1 - 0.66)

    img = np.squeeze(np.stack((R[np.newaxis, :, :], G[np.newaxis, :, :], B[np.newaxis, :, :]), axis=1))

    img[img < 0] = 0
    img[img < 0] = 0

    retImg = sitk.GetImageFromArray(img, isVector=False)
    dir = np.eye(3)
    olddir = np.array(Phalloidin.GetDirection()).reshape(2, -1)
    dir[0, :] = list(olddir[0, :]) + [0]
    dir[1, :] = list(olddir[1, :]) + [0]
    retImg.SetDirection(dir.flatten())
    retImg.SetSpacing(list(Phalloidin.GetSpacing()) + [1])
    retImg.SetOrigin(list(Phalloidin.GetOrigin()) + [0])

    return retImg
#####################

#     def Mic2PAS(PAS: sitk.Image) -> sitk.Image:
          
#           msk = getMask(PAS)
#           npPAS = sitk.GetArrayFromImage(PAS).astype(np.float32)
#           p2, p98 = np.percentile(npPAS[sitk.GetArrayViewFromImage(msk)!=0], (.005, 99.3))
#           npPAS = rescale_intensity(npPAS, in_range=(p2, p98), out_range=(0,1))
#          
#           R = 1 - npPAS * (1 - 0.88)
#           G = 1 - npPAS * (1 - 0.27)
#           B = 1 - npPAS * (1 - 0.66)

#           img = np.squeeze(np.stack((R[np.newaxis, :, :], G[np.newaxis, :, :], B[np.newaxis, :, :]), axis=1))        
#           img[img < 0] = 0
#           retImg = sitk.GetImageFromArray(img, isVector=False)

#           dir = np.eye(3)
#           olddir = np.array(PAS.GetDirection()).reshape(2, -1)
#           dir[0, :] = list(olddir[0, :]) + [0]
#           dir[1, :] = list(olddir[1, :]) + [0]
#           retImg.SetDirection(dir.flatten())
#           retImg.SetSpacing(list(PAS.GetSpacing()) + [1])
#           retImg.SetOrigin(list(PAS.GetOrigin()) + [0])
################################################################################################################


######### NZ: No modifications are required for PAS stain ##################################
def removeFiducialMarkersAlpha(locAlpha):

    locAlphaMask = sitk.OtsuThreshold(sitk.Median(locAlpha, (5, 5)), 0, 1)
    locAlphaMask = sitk.BinaryDilate(locAlphaMask, 10)
    locAlpha = sitk.Cast(((-1 * locAlphaMask) + 1), locAlpha.GetPixelID()) * locAlpha

    return locAlpha
############################################################################################


######### NZ: No modifications are required for PAS stain #########################################################
def getObjectBoundingBoxInPhysicalCoordinates(boundingBoxes, refImage: sitk.Image):         # NZ: I should expect boundingBoxes as an array
    nBoundingBoxes = len(boundingBoxes)
    physicalBoundingboxes = []


    for boundingBox in boundingBoxes:
        locBoundingBox = np.zeros(4)
        index1, index2 = refImage.TransformIndexToPhysicalPoint((int(boundingBox[0]), int(boundingBox[1])))
        locBoundingBox[0] = index1
        locBoundingBox[1] = index2

        index1, index2 = refImage.TransformIndexToPhysicalPoint((int(boundingBox[2]), int(boundingBox[3])))
        locBoundingBox[2] = index1
        locBoundingBox[3] = index2

        physicalBoundingboxes.append(list(locBoundingBox))

    return np.array(physicalBoundingboxes)

def getObjectBoundingBoxInPixelCoordinates(boundingBoxes, refImage: sitk.Image):
    nBoundingBoxes = len(boundingBoxes)
    physicalBoundingboxes = []

    for boundingBox in boundingBoxes:
        locBoundingBox = np.zeros(4, dtype=int)
        index1, index2 = refImage.TransformPhysicalPointToIndex((boundingBox[1], boundingBox[0]))
        locBoundingBox[0] = index1
        locBoundingBox[1] = index2

        index1, index2 = refImage.TransformPhysicalPointToIndex((boundingBox[3], boundingBox[2]))
        locBoundingBox[2] = index1
        locBoundingBox[3] = index2

        physicalBoundingboxes.append(list(locBoundingBox))

    return physicalBoundingboxes

def getObjectBoundingBoxes(img: sitk.Image, numSections=1) -> np.ndarray:  # NZ: changed the number of sections from 3 to 1

    img = sitk.SmoothingRecursiveGaussian(img, (15, 15, 1), True)
    msk = sitk.OtsuThreshold(img, 0, 1)
    cc: sitk.Image = sitk.ConnectedComponent(msk)

    cnts = np.bincount(sitk.GetArrayViewFromImage(cc).flatten())[1:]
    keeps = np.argsort(cnts)[-numSections:] + 1

    filt = sitk.LabelShapeStatisticsImageFilter()
    filt.Execute(cc)


    boundingBoxes = []
    for keep in keeps:
        bx = list(filt.GetBoundingBox(int(keep)))
        bx[0] -= 10  # startx
        bx[1] -= 10  # starty
        bx[2] += 20  # sizex
        bx[3] += 20  # sizey

        startxyReal = cc.TransformIndexToPhysicalPoint((bx[0], bx[1]))
        stopx = cc.TransformIndexToPhysicalPoint((bx[0] + bx[2], bx[1]))
        stopy = cc.TransformIndexToPhysicalPoint((bx[0], bx[1] + bx[3]))

        boundingBoxes.append([startxyReal, stopx, stopy])
    return boundingBoxes


def boundingBoxPhyscialToIndex(boundingBox: list[float], img: sitk.Image):
    startxyIx = img.TransformPhysicalPointToIndex(boundingBox[0])
    stopx = img.TransformPhysicalPointToIndex(boundingBox[1])
    stopy = img.TransformPhysicalPointToIndex(boundingBox[2])
    boundingBoxPixels = [
        startxyIx[0],
        startxyIx[1],
        stopx[0] - startxyIx[0],
        stopy[1] - startxyIx[1],
    ]

    return boundingBoxPixels


def applyObjectBounsdingBoxes(img: sitk.Image, boundingBoxes) -> list[sitk.Image]:

    npImg = sitk.GetArrayFromImage(img)

    imgList = []
    for box in boundingBoxes:

        boxLoc = boundingBoxPhyscialToIndex(box, img)

        npImgClipped = npImg[boxLoc[1] : boxLoc[1] + boxLoc[3], boxLoc[0] : boxLoc[0] + boxLoc[2]]


        cutImg = sitk.GetImageFromArray(npImgClipped)
        cutImg.SetSpacing(img.GetSpacing())
        cutImg.SetDirection(img.GetDirection())
        newOrigin = box[0]
        cutImg.SetOrigin(newOrigin)
        imgList.append(cutImg)

    return imgList


def saveImageList(imgList: list[sitk.Image], name: str, ouputdir: str) -> None:

    for ix, img in enumerate(imgList):
        sitk.WriteImage(img, pjoin(ouputdir, f"{name}_{ix}.mhd"))
##############################################################################################################################################


############## NZ: changes required for PAS stain ############################################################################################
def createComposites(phaloidinCuts: list[sitk.Image], DAPICuts: list[sitk.Image], alphaImgCuts: list[sitk.Image], outputDir: str,) -> None:

    for ix, (phaloidin, DAPI, alphaImg) in enumerate(zip(phaloidinCuts, DAPICuts, alphaImgCuts)):

        HE = Mic2HE(DAPI, phaloidin)

        sitk.WriteImage(HE, pjoin(outputDir, f"HE_{ix}.tif"))
        sitk.WriteImage(HE, pjoin(outputDir, f"HE_{ix}.nii"))

        npHE = sitk.GetArrayFromImage(HE).T
        npHE = np.swapaxes(npHE, 0, 1)

        alphaImg = sitk.Resample(alphaImg, DAPI)
        # sitk.WriteImage(alphaImg, pjoin(outputDir, f'alphaImgOverlay_{ix}.nii'))

        # Plot with contours [cut outs]
        npAlphaImg = sitk.GetArrayFromImage(alphaImg)

        fig, ax = plt.subplots()
        fig.set_size_inches((10, 10))
        plt.axis("off")
        im = ax.imshow(npHE, cmap = 'Grays')
        levels = np.linspace(np.percentile(npAlphaImg, 50), np.percentile(npAlphaImg, 98), 25)
        CS = ax.contourf(npAlphaImg, levels, linewidths=0.1, alpha = 0.2, cmap='jet')

        CB = fig.colorbar(CS, shrink=0.4, format=tkr.FormatStrFormatter('%.2g'), pad=0.03)
        l, b, w, h = ax.get_position().bounds
        ll, bb, ww, hh = CB.ax.get_position().bounds
        CB.ax.set_position([ll, b + 0.1 * h, ww, h * 0.8])
        CB.set_label('Activity [Bq]', rotation=270, labelpad=15 )
        plt.tight_layout()
        plt.savefig(pjoin(outputDir, f"_Contours_{ix}.tif"), dpi=600)

        plt.figure(figsize=(10, 10))
        plt.axis("off")
        npAlphaImg -= npAlphaImg.min()
        npAlphaImg /= np.percentile(npAlphaImg, 99.9)
        rgbAlphaImg = plt.cm.inferno(np.log(npAlphaImg + 1)).astype(np.float32)
        mixedPic = rgbAlphaImg[:, :, :3] * 1 + npHE.astype(np.float32) * 0.7

        plt.imshow(mixedPic, interpolation=None)
        plt.savefig(pjoin(outputDir, f"_Overlay_{ix}.tif"), dpi=600)

        plt.figure(figsize=(10, 10))
        plt.axis("off")
        plt.imshow(npHE, interpolation=None)
        plt.savefig(pjoin(outputDir, f"_HE_{ix}.tif"), dpi=600)


########################
#  def createComposites(PASCuts: list[sitk.Image], alphaImgCuts: list[sitk.Image], outputDir: str,) -> None:

#   for ix, (PAS, alphaImg) in enumerate(zip(PASCuts, alphaImgCuts)):
  
#       PasStain = Mic2PAS(PAS)
#       sitk.WriteImage(PasStain, pjoin(outputDir, f"PasStain_{ix}.tif"))
#       sitk.WriteImage(PasStain, pjoin(outputDir, f"PasStain_{ix}.nii"))
#       
#       nPasStain = sitk.GetArrayFromImage(PasStain).T
#       nPasStain = np.swapaxes(nPasStain, 0, 1)

#       alphaImg = sitk.Resample(alphaImg, PAS)      #?
#       npAlphaImg = sitk.GetArrayFromImage(alphaImg)
#       
#       fig, ax = plt.subplots()
#       fig.set_size_inches((10, 10))
#       plt.axis("off")
#       im = ax.imshow(nPasStain, cmap = 'Grays')
#       levels = np.linspace(np.percentile(npAlphaImg, 50), np.percentile(npAlphaImg, 98), 25)
#       CS = ax.contourf(npAlphaImg, levels, linewidths=0.1, alpha = 0.2, cmap='jet')
#       
#       CB = fig.colorbar(CS, shrink=0.4, format=tkr.FormatStrFormatter('%.2g'), pad=0.03)
#       l, b, w, h = ax.get_position().bounds
#       ll, bb, ww, hh = CB.ax.get_position().bounds
#       CB.ax.set_position([ll, b + 0.1 * h, ww, h * 0.8])
#       CB.set_label('Activity [Bq]', rotation=270, labelpad=15 )
#       plt.tight_layout()
#       plt.savefig(pjoin(outputDir, f"_Contours_{ix}.tif"), dpi=600)

#       plt.figure(figsize=(10, 10))
#       plt.axis("off")
#       npAlphaImg -= npAlphaImg.min()
#       npAlphaImg /= np.percentile(npAlphaImg, 99.9)
#       rgbAlphaImg = plt.cm.inferno(np.log(npAlphaImg + 1)).astype(np.float32)
#       mixedPic = rgbAlphaImg[:, :, :3] * 1 + nPasStain.astype(np.float32) * 0.7


#       plt.imshow(mixedPic, interpolation=None)
#       plt.savefig(pjoin(outputDir, f"_Overlay_{ix}.tif"), dpi=600)

#       plt.figure(figsize=(10, 10))
#       plt.axis("off")
#       plt.imshow(npPasStain, interpolation=None)
#       plt.savefig(pjoin(outputDir, f"_PasStain_{ix}.tif"), dpi=600)

########################################################################################################################################################