from os.path import join as pjoin

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import fill_voids
from skimage.exposure import equalize_adapthist, rescale_intensity
# color norm. standard (from TCGA-A2-A3XS-DX1, Amgad et al, 2019)
cnorm = {
    "mu": np.array([8.74108109, -0.12440419, 0.0444982]),
    "sigma": np.array([0.6135447, 0.10989545, 0.0286032]),
}

stain_unmixing_routine_params = {
    "stains": ["hematoxylin", "eosin"],
    "stain_unmixing_method": "macenko_pca",
}


# TCGA-A2-A3XS-DX1_xmin21421_ymin37486_.png, Amgad et al, 2019)
# for macenco (obtained using rgb_separate_stains_macenko_pca()
# and reordered such that columns are the order:
# Hamtoxylin, Eosin, Null
W_target = np.array(
    [
        [0.5807549, 0.08314027, 0.08213795],
        [0.71681094, 0.90081588, 0.41999816],
        [0.38588316, 0.42616716, -0.90380025],
    ]
)


def getMask(img: sitk.Image) -> sitk.Image:

    imggilt = sitk.SmoothingRecursiveGaussian(img, (15, 15, 1), True)
    npmsk = sitk.GetArrayFromImage(sitk.OtsuThreshold(imggilt, 0, 1)).astype(np.uint8)
    npmsk = fill_voids.fill(npmsk)

    msk = sitk.GetImageFromArray(npmsk)
    msk.CopyInformation(imggilt)

    return msk


def normalizeImageRange(
    img: sitk.Image, msk: sitk.Image, range=(0.1, 99.9)
) -> sitk.Image:

    npImg = sitk.GetArrayFromImage(img)
    npMsk = sitk.GetArrayFromImage(msk)

    npImg = npImg - np.percentile(npImg[npMsk != 0], range[0])
    npImg = npImg / np.percentile(npImg[npMsk != 0], range[1])
    npImg[npImg < 0] = 0
    # npImg[npImg > 1] = 1

    # npImg[msk==0] = 0

    retImg = sitk.GetImageFromArray(npImg)
    retImg.CopyInformation(img)

    return retImg


def resizeImg(img, new_size, interpolator=None) -> sitk.Image:
    # The spatial definition of the images we want to use in a deep learning framework (smaller than the original).
    reference_image = sitk.Image(new_size, img.GetPixelIDValue())
    reference_image.SetOrigin(img.GetOrigin())
    reference_image.SetDirection(img.GetDirection())
    reference_image.SetSpacing(
        [
            sz * spc / nsz
            for nsz, sz, spc in zip(new_size, img.GetSize(), img.GetSpacing())
        ]
    )

    if interpolator is None:
        return sitk.Resample(img, reference_image)
    else:
        return sitk.Resample(img, reference_image, sitk.Transform(), interpolator)


def Mic2HE(DAPI: sitk.Image, Phalloidin: sitk.Image, micScalingFactor) -> sitk.Image:
    npNuclearStain = sitk.GetArrayFromImage(DAPI).astype(np.float32)
    npCytosolStain = sitk.GetArrayFromImage(Phalloidin).astype(np.float32)  # *1.25
    # p2, p98 = np.percentile(npNuclearStain[sitk.GetArrayViewFromImage(msk) != 0], (2,98))
    p2, p98 = np.percentile(npNuclearStain[sitk.GetArrayViewFromImage(msk)!=0], (2,98))
    npNuclearStain = rescale_intensity(npNuclearStain, in_range=(p2, p98))
    npNuclearStain = equalize_adapthist(npNuclearStain, clip_limit=.01)

    p2, p98 = np.percentile(npCytosolStain[sitk.GetArrayViewFromImage(msk)!=0], (2,98))
    npCytosolStain = rescale_intensity(npCytosolStain, in_range=(p2, p98))
    npCytosolStain = equalize_adapthist(npCytosolStain, clip_limit=.01)

    # p2, p98 = np.percentile(npCytosolStain[sitk.GetArrayViewFromImage(msk)!=0], (2,98))
    # npCytosolStain = rescale_intensity(npCytosolStain, in_range=(p2, p98))

    # plt.figure()
    # plt.imshow(npNuclearStain)


    # npCytosolStain = npCytosolStain * sitk.GetArrayViewFromImage(msk)
    # npNuclearStain = npNuclearStain * sitk.GetArrayFromImage(msk)

    # gammaNuc = .75
    # gammaCyt = .85

    gammaNuc = 3
    gammaCyt = 1.1
    A, B = .7, .9

    npNuclearStain = A * npNuclearStain ** (gammaNuc)
    npCytosolStain = B * (npCytosolStain) ** (gammaCyt)

    R = 1 - npNuclearStain * (1 - 0.24) - npCytosolStain * (1 - 0.88)
    G = 1 - npNuclearStain * (1 - 0.21) - npCytosolStain * (1 - 0.27)
    B = 1 - npNuclearStain * (1 - 0.62) - npCytosolStain * (1 - 0.66)

    # R = 1 - npNuclearStain*(W_target[0,0]) - npCytosolStain*(W_target[0,1])
    # G = 1 - npNuclearStain*(W_target[1,0]) - npCytosolStain*(W_target[1,1])
    # B = 1 - npNuclearStain*(W_target[2,0]) - npCytosolStain*(W_target[2,1])

    img = np.squeeze(
        np.stack((R[np.newaxis, :, :], G[np.newaxis, :, :], B[np.newaxis, :,:]), axis=1)
    )

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


def removeFiducialMarkersAlpha(locAlpha):

    locAlphaMask = sitk.OtsuThreshold(sitk.Median(locAlpha, (5, 5)), 0, 1)
    locAlphaMask = sitk.BinaryDilate(locAlphaMask, 10)
    locAlpha = sitk.Cast(((-1 * locAlphaMask) + 1), locAlpha.GetPixelID()) * locAlpha

    return locAlpha


def getObjectBoundingBoxInPhysicalCoordinates(boundingBoxes, refImage: sitk.Image):
    nBoundingBoxes = len(boundingBoxes)
    physicalBoundingboxes = []

    for boundingBox in boundingBoxes:
        # This is always 4 points, groups of 2 for x-y
        locBoundingBox = np.zeros(4)
        index1, index2 = refImage.TransformIndexToPhysicalPoint(
            (int(boundingBox[0]), int(boundingBox[1]))
        )
        locBoundingBox[0] = index1
        locBoundingBox[1] = index2

        index1, index2 = refImage.TransformIndexToPhysicalPoint(
            (int(boundingBox[2]), int(boundingBox[3]))
        )
        locBoundingBox[2] = index1
        locBoundingBox[3] = index2

        physicalBoundingboxes.append(list(locBoundingBox))

    return np.array(physicalBoundingboxes)


def getObjectBoundingBoxInPixelCoordinates(boundingBoxes, refImage: sitk.Image):
    nBoundingBoxes = len(boundingBoxes)
    physicalBoundingboxes = []

    for boundingBox in boundingBoxes:
        # This is always 4 points, groups of 2 for x-y
        locBoundingBox = np.zeros(4, dtype=int)
        index1, index2 = refImage.TransformPhysicalPointToIndex(
            (boundingBox[1], boundingBox[0])
        )
        locBoundingBox[0] = index1
        locBoundingBox[1] = index2

        index1, index2 = refImage.TransformPhysicalPointToIndex(
            (boundingBox[3], boundingBox[2])
        )
        locBoundingBox[2] = index1
        locBoundingBox[3] = index2

        physicalBoundingboxes.append(list(locBoundingBox))

    return physicalBoundingboxes


def getObjectBoundingBoxes(img: sitk.Image, numSections=3) -> np.ndarray:

    img = sitk.SmoothingRecursiveGaussian(img, (15, 15, 1), True)
    msk = sitk.OtsuThreshold(img, 0, 1)
    cc: sitk.Image = sitk.ConnectedComponent(msk)

    cnts = np.bincount(sitk.GetArrayViewFromImage(cc).flatten())[1:]
    keeps = np.argsort(cnts)[-numSections:] + 1

    filt = sitk.LabelShapeStatisticsImageFilter()
    filt.Execute(cc)

    # extract bounding boxes
    boundingBoxes = []
    for keep in keeps:
        bx = list(filt.GetBoundingBox(int(keep)))
        bx[0] -= 10  # startx
        bx[1] -= 10  # starty
        bx[2] += 20  # sizex
        bx[3] += 20  # sizey

        # translate boundingbox coordinates into real space
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

        npImgClipped = npImg[
            boxLoc[1] : boxLoc[1] + boxLoc[3], boxLoc[0] : boxLoc[0] + boxLoc[2]
        ]

        plt.figure()
        plt.imshow(npImgClipped)
        plt.savefig('tmpout.png')

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


def createComposites(
    phaloidinCuts: list[sitk.Image],
    DAPICuts: list[sitk.Image],
    alphaImgCuts: list[sitk.Image],
    micScalingFactor,
    outputDir: str,
) -> None:

    for ix, (phaloidin, DAPI, alphaImg) in enumerate(
        zip(phaloidinCuts, DAPICuts, alphaImgCuts)
    ):

        msk = getMask(DAPI)
        phaloidin = normalizeImageRange(phaloidin, msk, range=(0.1, 98))
        DAPI = normalizeImageRange(DAPI, msk)

        HE = Mic2HE(DAPI, phaloidin, micScalingFactor)

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
        im = ax.imshow(npHE)
        levels = np.linspace(npAlphaImg.min(), np.percentile(npAlphaImg, 90), 10)
        CS = ax.contour(npAlphaImg, levels, linewidths=0.5)

        CB = fig.colorbar(CS, shrink=0.8)
        l, b, w, h = ax.get_position().bounds
        ll, bb, ww, hh = CB.ax.get_position().bounds
        CB.ax.set_position([ll, b + 0.1 * h, ww, h * 0.8])
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
