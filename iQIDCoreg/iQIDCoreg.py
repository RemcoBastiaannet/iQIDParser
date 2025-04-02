import os
from os.path import join as pjoin

import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import pickle
import copy

def correctResolutionTransform(transforms: list, moving: sitk.Image):

    superResTrans = []
    for trans in transforms:
        newSize = [
            int(
                np.ceil(
                    float(trans["Size"][i])
                    / (moving.GetSpacing()[i] / float(trans["Spacing"][i]))
                )
            )
            for i in range(2)
        ]
        newSpacing = moving.GetSpacing()
        trans["Spacing"] = [str(i) for i in newSpacing]
        trans["Size"] = [str(i) for i in newSize]

        superResTrans.append(trans)

    return superResTrans


coords = [[], []]


def getCorrespondingSections(alpha: sitk.Image, micImage: sitk.Image, numSectionsPerSlide=3):
    
    global coords
    coords = [[], []]

    alphaImg = sitk.GetArrayFromImage(alpha)
    alphaImg = alphaImg - alphaImg.min()

    alphaPlot = np.log(alphaImg+1)
    mx = np.percentile(alphaPlot[alphaPlot > 0], 99.9)

    mx2 = np.percentile(sitk.GetArrayViewFromImage(micImage), 99.9)
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].imshow(sitk.GetArrayFromImage(micImage), vmax = mx2, cmap='Reds')
    ax[0].axis('off')

    ax[1].imshow(alphaPlot, vmax=mx, cmap='inferno')
    ax[1].axis('off')

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()

    def onclick(event):
        global coords
        # Only clicks inside this axis are valid.

        # try:  # use try/except in case we are not using Qt backend
        #     # 0 is the arrow, which means we are not zooming or panning.

        #     zooming_panning = (fig.canvas.cursor().shape() != Cursors(1))
        # except:
        #     zooming_panning = False
        # if zooming_panning:
        #     print("Panning and zooming")
        #     return

        if fig.canvas.widgetlock.locked():
            print("Panning and zooming")
            return

        global ix, iy
        ix, iy = event.xdata, event.ydata
        print(f'x = {ix}, y = {iy}')

        for ixx, a in enumerate(ax):
            if a == event.inaxes:

                coords[ixx].append((ix, iy))

                a.scatter(ix, iy)
                a.figure.canvas.draw()
                # fig.canvas.draw()

        if len(coords[0]) == numSectionsPerSlide and len(coords[1]) == numSectionsPerSlide:
            fig.canvas.mpl_disconnect(cid)
            plt.close(fig)

            return coords

    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()

    return coords


def matchPointClouds(coords: list[list], alphaImg: sitk.Image, MicImg: sitk.Image) -> sitk.Image:
    # Get corresponding points for each image
    # np.array([[5099, 25350], [4766, 15454], [5233, 5257]])
    pointsMic = np.array(coords[0])
    # [[1932, 648], [1421, 977], [874, 1240]]) / 2.
    pointsAlpha = np.array(coords[1])

    PhysicalPointsAlpha = np.array(
        [i * alphaImg.GetSpacing() for i in pointsAlpha]).T
    PhysicalPointsMic = np.array(
        [i * MicImg.GetSpacing() for i in pointsMic]).T

    p1 = PhysicalPointsAlpha #moving img
    p2 = PhysicalPointsMic
    # Calculate centroids
    # If you don't put reshape then the outcome is 1D with no rows/colums and is interpeted as rowvector in next minus operation, while it should be a column vector
    p1_c = np.mean(p1, axis=1).reshape((-1, 1))
    p2_c = np.mean(p2, axis=1).reshape((-1, 1))

    # Subtract centroids
    q1 = p1-p1_c
    q2 = p2-p2_c

    # Calculate covariance matrix
    H = np.matmul(q1, q2.transpose())

    # Calculate singular value decomposition (SVD)
    U, X, V_t = np.linalg.svd(H)  # the SVD of linalg gives you Vt

    # Calculate rotation matrix
    R = np.matmul(V_t.transpose(), U.transpose())

    # assert np.allclose(np.linalg.det(
    #     R), 1.0), "Rotation matrix of N-point registration not 1, see paper Arun et al."

    # Calculate translation matrix
    T = p2_c - np.matmul(R, p1_c)

    result = T + np.matmul(R, p1)

    # Calculate new direction matrix and origin
    alpha2 = copy.copy(alphaImg)
    alpha2.SetDirection(R.flatten())
    alpha2.SetOrigin([i for i in T.flatten()])

    return alpha2

def coregAlphaCameraImgToMicImg(DAPIImg: sitk.Image, phalloidinImg: sitk.Image, micImg: sitk.Image):
    parMap = sitk.GetDefaultParameterMap("rigid")
    parMap['AutomaticTransformInitialization'] = ('false',)
    parMap['Transform'] = ('SimilarityTransform',)
    # parMap['ImageSampler'] = ('RandomSparseMask',)
    parMap['MaximumNumberOfIterations'] = ("2580",)
    # parMap['NumberOfResolutions'] = ('1',)
    # parMap['Metric'] = ('AdvancedNormalizedCorrelation',)

    # parMap['NumberOfSpatialSamples'] = (
    #     str(int(np.prod(phalloidinImg.GetSize()) * 0.1)),)
    parMap['ASGDParameterEstimationMethod'] = ("DisplacementDistribution",)


    parMap2 = sitk.GetDefaultParameterMap("rigid")
    # parMap2['Transform'] = ('SimilarityTransform',)
    parMap2['Transform'] = ('SimilarityTransform',)
    # parMap2['Metric'] = ('AdvancedNormalizedCorrelation',)
    parMap2['AutomaticTransformInitialization'] = ('false',)
    # parMap2['ImageSampler'] = ('RandomSparseMask',)
    # parMap2['NumberOfResolutions'] = ('1',)
    parMap2['ASGDParameterEstimationMethod'] = ("DisplacementDistribution",)
    # parMap2['NumberOfSpatialSamples'] = (
    #     str(int(np.prod(phalloidinImg.GetSize()) * 0.10)),)
    parMap2['MaximumNumberOfIterations'] = ("4000",)
    # parMap2['CheckNumberOfSamples'] = ('false',)

    parMap3 = sitk.GetDefaultParameterMap('bspline')
    parMap3['NumberOfResolutions']= ("1",)
    # parMap3['GridSpacingSchedule'] = (str(1.4 * alphaImg.GetSpacing()[0]), str(1.0 * alphaImg.GetSpacing()[0]))
    parMap3['GridSpacingSchedule'] = (str(1.0 * phalloidinImg.GetSpacing()[0]),)
    parMap3['ASGDParameterEstimationMethod'] = ("DisplacementDistribution",)
    parMap3['FinalGridSpacingInPhysicalUnits'] = (str(18*micImg.GetSpacing()[0]),)
    parMap3['MovingImagePyramid'] = ("MovingShrinkingImagePyramid",)
    parMap3['FixedImagePyramid'] = ("FixedShrinkingImagePyramid",)
    parMap3['Metric'] = ("AdvancedMattesMutualInformation","TransformBendingEnergyPenalty")
    # parMap3['Metric'] = ("AdvancedNormalizedCorrelation","TransformBendingEnergyPenalty")
    parMap3['Metric1Weight'] = ("1.5",)
    parMap3['MaximumNumberOfIterations'] = ("2000",)

    parMaps = sitk.VectorOfParameterMap()
    parMaps.append(parMap)
    parMaps.append(parMap2)
    # parMaps.append(parMap3)

    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetParameterMap(parMaps)

    ##Create mask of only the overlapping bit
    # micImgInAlpha = sitk.Resample(micImg, alphaImg, sitk.Transform(), sitk.sitkBSpline)
    #but now, this image has too many pixels / too small pixel spacing
    
    newSize = [phalloidinImg.GetSpacing()[i]/micImg.GetSpacing()[i] * phalloidinImg.GetSize()[i] for i in range(micImg.GetDimension())]
    newSize = [int(np.ceil(i)) for i in newSize]
    refImage = sitk.Image(*newSize, micImg.GetPixelID())
    refImage.SetOrigin(phalloidinImg.GetOrigin())
    refImage.SetDirection(phalloidinImg.GetDirection())
    refImage.SetSpacing(micImg.GetSpacing())

    micImgInAlpha = sitk.Resample(micImg, refImage, sitk.Transform(), sitk.sitkNearestNeighbor)

    # micImgInAlphaMask = micImgInAlpha > 1E-4
    # micImgInAlphaMask = sitk.BinaryDilate(micImgInAlphaMask, 2)
    # sitk.WriteImage(micImgInAlphaMask, 'micImgInAlphaMask.nii')

    # alphaImgMask = sitk.OtsuThreshold(alphaImg, 0, 1)
    # alphaImgMask = sitk.BinaryDilate(alphaImgMask, 5)
    # sitk.WriteImage(alphaImgMask, 'alphaImgMask.nii')
    # sitk.WriteImage(micImgInAlpha, "AlphaResampToMic.mhd")

    # sitk.WriteImage(alphaImg, 'micImageForcorreg.mhd')
    # alphaMaskInMic = sitk.Cast(sitk.OtsuThreshold(alphaInMic, 0, 1), alphaInMic.GetPixelID())
    # alphaMask = sitk.Resample(alphaMaskInMic, alphaImg)

    # tmpAlpha = alphaImg * alphaMask
    
    # sitk.WriteImage(tmpAlpha, 'alphaMask.nii')
    # sitk.WriteImage(tmpAlpha, 'maskedAlpha.nii')

    # elastixImageFilter.SetMovingImage(alphaImg)
    # elastixImageFilter.SetFixedImage(micImgInAlpha)
    elastixImageFilter.SetMovingImage(phalloidinImg)
    elastixImageFilter.SetFixedImage(micImgInAlpha)
    # elastixImageFilter.SetFixedMask(micImgInAlphaMask)
    # elastixImageFilter.SetMovingMask(alphaImgMask)
    # elastixImageFilter.SetMovingMask(sitk.HuangThreshold(alphaImg, 0, 1))
    # elastixImageFilter.SetFixedMask(sitk.TriangleThreshold(micImgInAlpha, 0, 1))
    # elastixImageFilter.SetMovingMask(sitk.Cast(alphaImg>0, sitk.sitkUInt8))
    # elastixImageFilter.SetFixedMask(sitk.Cast(micImg>0, sitk.sitkUInt8))
    # elastixImageFilter.SetFixedImage(micImgInAlpha)
    elastixImageFilter.SetLogToFile(False)
    elastixImageFilter.SetLogToConsole(True)
    elastixImageFilter.Execute()
    sitk.WriteImage(elastixImageFilter.GetResultImage(), 'localresult.nii')

    # transforms = elastixImageFilter.GetTransformParameterMap()
    numTransforms = 2#len([i for i in os.listdir('.') if r'TransformParameters.' in i ])
    transforms = [sitk.ReadParameterFile(f'TransformParameters.{i}.txt') for i in range(numTransforms)]#,sitk.ReadParameterFile('TransformParameters.1.txt')]
    #making sure initial tansfroms are correct
    for trans in transforms:
        trans['InitialTransformParametersFileName'] =  ("NoInitialTransform",)

    newTransforms = correctResolutionTransform(transforms, DAPIImg)

    alphaImgTrans = sitk.Transformix(DAPIImg, newTransforms)
    alphaImg2Trans = sitk.Transformix(phalloidinImg, newTransforms)
    return alphaImgTrans, alphaImg2Trans


def anatomicalCoreg(fixed: sitk.Image, moving: sitk.Image) -> tuple[sitk.Image, sitk.ParameterMap]:
    # parMap = sitk.GetDefaultParameterMap("translation")
    # parMap['AutomaticTransformInitialization'] = ('false',)
    # parMap['ImageSampler'] = ('RandomSparseMask',)

    parMap2 = sitk.GetDefaultParameterMap("rigid")
    # parMap2['AutomaticTransformInitialization'] = ('true',)
    # parMap2['AutomaticTransformInitializationMethod'] = ("CenterOfMass",)
    parMap2['ASGDParameterEstimationMethod'] = ("DisplacementDistribution",)
    # parMap2['ImageSampler'] = ('RandomSparseMask',)
    parMap2['MaximumNumberOfIterations'] = ("500",)
    parMap2['NumberOfSpatialSamples'] = (
        str(int(np.prod(fixed.GetSize()) * 0.005)),)

    parMap3 = sitk.GetDefaultParameterMap("bspline")
    # parMap3['AutomaticTransformInitialization'] = ('false',)
    parMap3['ASGDParameterEstimationMethod'] = ("DisplacementDistribution",)
    parMap3['FinalGridSpacingInPhysicalUnits'] = ('150',)
    parMap3['NumberOfSpatialSamples'] = (
        str(int(np.prod(fixed.GetSize()) * 0.005)),)
    # parMap3['ImageSampler'] = ('RandomSparseMask',)

    parMaps = sitk.VectorOfParameterMap()
    # parMaps.append(parMap)
    parMaps.append(parMap2)
    parMaps.append(parMap3)

    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetParameterMap(parMaps)
    elastixImageFilter.SetMovingImage(moving)
    # elastixImageFilter.SetMovingMask(sitk.Cast(phaloidinCuts[1]>0., sitk.sitkUInt8))
    # elastixImageFilter.SetFixedMask(sitk.Cast(phaloidinCuts[0] > 0., sitk.sitkUInt8))
    elastixImageFilter.SetFixedImage(fixed)
    elastixImageFilter.SetLogToFile(True)
    elastixImageFilter.SetLogToConsole(True)
    elastixImageFilter.Execute()
    movingInFixed = elastixImageFilter.GetResultImage()
    transform = elastixImageFilter.GetTransformParameterMap()
    return movingInFixed, transform


def doFullAnatomicalCoreg(phaloidinCuts: list[sitk.Image], DAPICuts: list[sitk.Image], alphaImgCuts: list[sitk.Image], outputDir: str) -> None:

    for ixImg, phaloidin in enumerate(phaloidinCuts):
        if ixImg == 0:
            sitk.WriteImage(phaloidinCuts[ixImg], pjoin(
                outputDir, f'Coreg_phaloidinCuts{ixImg}.mhd'))
            sitk.WriteImage(DAPICuts[ixImg], pjoin(
                outputDir, f'Coreg_DAPICuts_{ixImg}.mhd'))
            sitk.WriteImage(alphaImgCuts[ixImg], pjoin(
                outputDir, f'Coreg_alphaImgCuts_{ixImg}.mhd'))
            continue

        img, transform = anatomicalCoreg(phaloidinCuts[0], phaloidin)

        phaloidinCuts[ixImg] = img
        DAPICuts[ixImg] = sitk.Transformix(DAPICuts[ixImg], transform)
        transform = correctResolutionTransform(transform, alphaImgCuts[ixImg])
        alphaImgCuts[ixImg] = sitk.Transformix(alphaImgCuts[ixImg], transform)

        sitk.WriteImage(phaloidinCuts[ixImg], pjoin(
            outputDir, f'Coreg_phaloidinCuts{ixImg}.mhd'))
        sitk.WriteImage(DAPICuts[ixImg], pjoin(
            outputDir, f'Coreg_DAPICuts_{ixImg}.mhd'))
        sitk.WriteImage(alphaImgCuts[ixImg], pjoin(
            outputDir, f'Coreg_alphaImgCuts_{ixImg}.mhd'))

def wrapperGetCorrespondingSections(alphaImg: sitk.Image, micImage: sitk.Image, outputDir: str, ignoreExist = False) -> list[list]:

    if not ignoreExist and os.path.isfile(pjoin(outputDir, 'matchingCoords.pic')):
        with open(pjoin(outputDir, 'matchingCoords.pic'), 'rb') as f:
            coords = pickle.load(f)

    else:
        coords = getCorrespondingSections(alphaImg, micImage)
        print(f"Coords: {coords}")

        with open(pjoin(outputDir, 'matchingCoords.pic'), 'wb') as f:
            pickle.dump(coords, f)
    
    return coords


def doFullCoreg(alphaImg: sitk.Image,  dapiImage: sitk.Image, phaloidinImage: sitk.Image, matchingCoords: list[list], outputDir: str) -> sitk.Image:

    matchingCoords = matchingCoords[::-1]
    dapiImage = matchPointClouds(matchingCoords, dapiImage, alphaImg)
    phaloidinImage = matchPointClouds(matchingCoords, phaloidinImage, alphaImg)

    sitk.WriteImage(dapiImage, pjoin(
        outputDir, 'micDAPIImgPointCloudMatch.mhd'))
    sitk.WriteImage(phaloidinImage, pjoin(
        outputDir, 'micPHALImgPointCloudMatch.mhd'))
    # alphaImg = sitk.Resample(alphaImg, dapiImage)
    # sitk.WriteImage(alphaImg, pjoin(
    #     outputDir, 'alphaResampToPhaloidin.mhd'))
    # alphaImg = sitk.Resample(alphaImg, dapiImage, sitk.Transform(), sitk.sitkBSpline)
    # sitk.WriteImage(alphaImg, pjoin(outputDir, 'alphaCameraImg_resamp.nii'))

    alphaImg = sitk.Median(alphaImg, (3,3))
    # sitk.WriteImage(alphaImg, 'alphafilt.nii')
    dapiImageInAlpha, phaloidinInAlpha = coregAlphaCameraImgToMicImg(dapiImage, phaloidinImage, alphaImg)
    # sitk.WriteImage(alphaImg, 'alphatmp.mhd')
    # sitk.WriteImage(alphaImg, pjoin(
    #     outputDir, 'fullAlphaImageInPhaloidin.mhd'))

    sitk.WriteImage(dapiImageInAlpha, pjoin(
        outputDir, '_fullDAPIImageInAlpha.mhd'))
    sitk.WriteImage(phaloidinInAlpha, pjoin(
        outputDir, '_fullPhaloidinImageInAlpha.mhd'))

    return dapiImageInAlpha, phaloidinInAlpha
