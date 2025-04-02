import os
from os.path import join as pjoin

import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import pickle


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



############### NZ: required changes for PAS stain ######################################################################

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
##################################

#       def getCorrespondingSections(alpha: sitk.Image, micImage: sitk.Image, numSectionsPerSlide=3):
#       global coords
#       coords = [[], []]

#       alphaImg = sitk.GetArrayFromImage(alpha)
#       alphaImg = alphaImg - alphaImg.min()
#       alphaPlot = np.log(alphaImg+1)
#       mx = np.percentile(alphaPlot[alphaPlot > 0], 99.9)
#       mx2 = np.percentile(sitk.GetArrayViewFromImage(micImage), 99.9)
#       fig, ax = plt.subplots(nrows=1, ncols=2)
#       ax[0].imshow(sitk.GetArrayFromImage(micImage), vmax = mx2, cmap='Reds')
#       ax[0].axis('off')
#
#       ax[1].imshow(alphaPlot, vmax=mx, cmap='inferno')
#       ax[1].axis('off')

#       figManager = plt.get_current_fig_manager()
#       figManager.window.showMaximized()

#       figManager = plt.get_current_fig_manager()
#       figManager.window.showMaximized()

#          def onclick(event):
#              global coords

#              if fig.canvas.widgetlock.locked():
#              print("Panning and zooming")
#              return

#              global ix, iy
#              ix, iy = event.xdata, event.ydata
#              print(f'x = {ix}, y = {iy}')

#              for ixx, a in enumerate(ax):
#              if a == event.inaxes:

#              coords[ixx].append((ix, iy))

#              a.scatter(ix, iy)
#              a.figure.canvas.draw()
#             

#              if len(coords[0]) == numSectionsPerSlide and len(coords[1]) == numSectionsPerSlide:
#              fig.canvas.mpl_disconnect(cid)
#              plt.close(fig)

#              return coords

#              cid = fig.canvas.mpl_connect('button_press_event', onclick)

#              plt.show()

#     return coords
##################################################################################################################################



def matchPointClouds(coords: list[list], alphaImg: sitk.Image, MicImg: sitk.Image) -> sitk.Image:

    pointsMic = np.array(coords[0])

    pointsAlpha = np.array(coords[1])

    PhysicalPointsAlpha = np.array([i * alphaImg.GetSpacing() for i in pointsAlpha]).T

    PhysicalPointsMic = np.array([i * MicImg.GetSpacing() for i in pointsMic]).T

    p1 = PhysicalPointsAlpha
    p2 = PhysicalPointsMic

    p1_c = np.mean(p1, axis=1).reshape((-1, 1))
    p2_c = np.mean(p2, axis=1).reshape((-1, 1))

  
    q1 = p1-p1_c
    q2 = p2-p2_c

  
    H = np.matmul(q1, q2.transpose())

   
    U, X, V_t = np.linalg.svd(H)  

  
    R = np.matmul(V_t.transpose(), U.transpose())

 
    T = p2_c - np.matmul(R, p1_c)

    result = T + np.matmul(R, p1)

   
    alpha2 = alphaImg
    alpha2.SetDirection(R.flatten())
    alpha2.SetOrigin([i for i in T.flatten()])

    return alpha2


################ NZ: required changes for PAS stain ##############################################################

def coregAlphaCameraImgToMicImg(DAPIImg: sitk.Image, phalloidinImg: sitk.Image, micImg: sitk.Image):
    parMap = sitk.GetDefaultParameterMap("rigid")
    parMap['AutomaticTransformInitialization'] = ('false',)

    parMap['MaximumNumberOfIterations'] = ("2580",)
    parMap['NumberOfResolutions'] = ('1',)
    parMap['Metric'] = ('AdvancedNormalizedCorrelation',)




    parMap2 = sitk.GetDefaultParameterMap("rigid")
  
    parMap2['Transform'] = ('SimilarityTransform',)
    parMap2['Metric'] = ('AdvancedNormalizedCorrelation',)
    parMap2['AutomaticTransformInitialization'] = ('false',)
    parMap2['NumberOfResolutions'] = ('1',)
    parMap2['NumberOfSpatialSamples'] = (str(int(np.prod(phalloidinImg.GetSize()) * 0.10)),)
    parMap2['MaximumNumberOfIterations'] = ("4000",)

    parMap3 = sitk.GetDefaultParameterMap('bspline')
    parMap3['NumberOfResolutions']= ("1",)
    parMap3['GridSpacingSchedule'] = (str(1.0 * phalloidinImg.GetSpacing()[0]),)
    parMap3['ASGDParameterEstimationMethod'] = ("DisplacementDistribution",)
    parMap3['FinalGridSpacingInPhysicalUnits'] = (str(18*micImg.GetSpacing()[0]),)
    parMap3['MovingImagePyramid'] = ("MovingShrinkingImagePyramid",)
    parMap3['FixedImagePyramid'] = ("FixedShrinkingImagePyramid",)
    parMap3['Metric'] = ("AdvancedMattesMutualInformation","TransformBendingEnergyPenalty")
    parMap3['Metric1Weight'] = ("1.5",)
    parMap3['MaximumNumberOfIterations'] = ("2000",)

    parMaps = sitk.VectorOfParameterMap()
    parMaps.append(parMap)
    parMaps.append(parMap2)

    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetParameterMap(parMaps)


    
    newSize = [phalloidinImg.GetSpacing()[i]/micImg.GetSpacing()[i] * phalloidinImg.GetSize()[i] for i in range(micImg.GetDimension())]
    newSize = [int(np.ceil(i)) for i in newSize]
    refImage = sitk.Image(*newSize, micImg.GetPixelID())
    refImage.SetOrigin(phalloidinImg.GetOrigin())
    refImage.SetDirection(phalloidinImg.GetDirection())
    refImage.SetSpacing(micImg.GetSpacing())

    micImgInAlpha = sitk.Resample(micImg, refImage, sitk.Transform(), sitk.sitkNearestNeighbor)

    elastixImageFilter.SetMovingImage(phalloidinImg)
    elastixImageFilter.SetFixedImage(micImgInAlpha)

    elastixImageFilter.SetLogToFile(False)
    elastixImageFilter.SetLogToConsole(True)
    elastixImageFilter.Execute()
    sitk.WriteImage(elastixImageFilter.GetResultImage(), 'localresult.nii')

    
    numTransforms = 2
    transforms = [sitk.ReadParameterFile(f'TransformParameters.{i}.txt') for i in range(numTransforms)]
    for trans in transforms:
        trans['InitialTransformParametersFileName'] =  ("NoInitialTransform",)

    newTransforms = correctResolutionTransform(transforms, DAPIImg)

    alphaImgTrans = sitk.Transformix(phalloidinImg, newTransforms)
    alphaImg2Trans = sitk.Transformix(DAPIImg, newTransforms)
    return alphaImgTrans, alphaImg2Trans
###########################################

# def coregAlphaCameraImgToMicImg(PASImg: sitk.Image, micImg: sitk.Image):
#    parMap = sitk.GetDefaultParameterMap("rigid")
#    parMap['AutomaticTransformInitialization'] = ('false',)

#    parMap['MaximumNumberOfIterations'] = ("2580",)
#    parMap['NumberOfResolutions'] = ('1',)
#    parMap['Metric'] = ('AdvancedNormalizedCorrelation',)

#   parMap2 = sitk.GetDefaultParameterMap("rigid")
  
#    parMap2['Transform'] = ('SimilarityTransform',)
#    parMap2['Metric'] = ('AdvancedNormalizedCorrelation',)
#    parMap2['AutomaticTransformInitialization'] = ('false',)
#    parMap2['NumberOfResolutions'] = ('1',)
#    parMap2['NumberOfSpatialSamples'] = (str(int(np.prod(PASImg.GetSize()) * 0.10)),)
#    parMap2['MaximumNumberOfIterations'] = ("4000",)
#    parMap3 = sitk.GetDefaultParameterMap('bspline')
#    parMap3['NumberOfResolutions']= ("1",)
#    parMap3['GridSpacingSchedule'] = (str(1.0 * PASImg.GetSpacing()[0]),)
#    parMap3['ASGDParameterEstimationMethod'] = ("DisplacementDistribution",)
#    parMap3['FinalGridSpacingInPhysicalUnits'] = (str(18*micImg.GetSpacing()[0]),)
#    parMap3['MovingImagePyramid'] = ("MovingShrinkingImagePyramid",)
#    parMap3['FixedImagePyramid'] = ("FixedShrinkingImagePyramid",)
#    parMap3['Metric'] = ("AdvancedMattesMutualInformation","TransformBendingEnergyPenalty")
#    parMap3['Metric1Weight'] = ("1.5",)
#    parMap3['MaximumNumberOfIterations'] = ("2000",)

#    parMaps = sitk.VectorOfParameterMap()
#    parMaps.append(parMap)
#    parMaps.append(parMap2)

#    elastixImageFilter = sitk.ElastixImageFilter()
#    elastixImageFilter.SetParameterMap(parMaps)

#    newSize = [PASImg.GetSpacing()[i]/micImg.GetSpacing()[i] * PASImg.GetSize()[i] for i in range(micImg.GetDimension())]
#    newSize = [int(np.ceil(i)) for i in newSize]
#    refImage = sitk.Image(*newSize, micImg.GetPixelID())
#    refImage.SetOrigin(PASImg.GetOrigin())
#    refImage.SetDirection(PASImg.GetDirection())
#    refImage.SetSpacing(micImg.GetSpacing())

#    micImgInAlpha = sitk.Resample(micImg, refImage, sitk.Transform(), sitk.sitkNearestNeighbor)
#    elastixImageFilter.SetMovingImage(PASImg)
#    elastixImageFilter.SetFixedImage(micImgInAlpha)

#    elastixImageFilter.SetLogToFile(False)
#    elastixImageFilter.SetLogToConsole(True)
#    elastixImageFilter.Execute()
#    sitk.WriteImage(elastixImageFilter.GetResultImage(), 'localresult.nii')
#    
#   
#    transforms = sitk.ReadParameterFile(f'TransformParameters.0.txt')
#    transforms['InitialTransformParametersFileName'] =  ("NoInitialTransform",)

#    newTransforms = correctResolutionTransform(transforms, PASImg)  #?

#    alphaImgTrans = sitk.Transformix(PASImg, newTransforms)
#    return alphaImgTrans


#####################################################################################################################################


def anatomicalCoreg(fixed: sitk.Image, moving: sitk.Image) -> tuple[sitk.Image, sitk.ParameterMap]:
    parMap = sitk.GetDefaultParameterMap("translation")

    parMap2 = sitk.GetDefaultParameterMap("rigid")
    parMap2['AutomaticTransformInitialization'] = ('true',)
    parMap2['AutomaticTransformInitializationMethod'] = ("CenterOfMass",)
    parMap2['ASGDParameterEstimationMethod'] = ("DisplacementDistribution",)

    parMap2['MaximumNumberOfIterations'] = ("500",)
    parMap2['NumberOfSpatialSamples'] = (str(int(np.prod(fixed.GetSize()) * 0.005)),)

    parMap3 = sitk.GetDefaultParameterMap("bspline")
    parMap3['ASGDParameterEstimationMethod'] = ("DisplacementDistribution",)
    parMap3['FinalGridSpacingInPhysicalUnits'] = ('150',)
    parMap3['NumberOfSpatialSamples'] = (str(int(np.prod(fixed.GetSize()) * 0.005)),)

    parMaps = sitk.VectorOfParameterMap()
    parMaps.append(parMap2)
    parMaps.append(parMap3)

    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetParameterMap(parMaps)
    elastixImageFilter.SetMovingImage(moving)

    elastixImageFilter.SetFixedImage(fixed)
    elastixImageFilter.SetLogToFile(True)
    elastixImageFilter.SetLogToConsole(True)
    elastixImageFilter.Execute()
    movingInFixed = elastixImageFilter.GetResultImage()
    transform = elastixImageFilter.GetTransformParameterMap()
    return movingInFixed, transform



################ NZ: required changes for PAS stain ##############################################################################################
def doFullAnatomicalCoreg(phaloidinCuts: list[sitk.Image], DAPICuts: list[sitk.Image], alphaImgCuts: list[sitk.Image], outputDir: str) -> None:

    for ixImg, phaloidin in enumerate(phaloidinCuts):
        if ixImg == 0:
            sitk.WriteImage(phaloidinCuts[ixImg], pjoin(outputDir, f'Coreg_phaloidinCuts{ixImg}.mhd'))
            sitk.WriteImage(DAPICuts[ixImg], pjoin(outputDir, f'Coreg_DAPICuts_{ixImg}.mhd'))
            sitk.WriteImage(alphaImgCuts[ixImg], pjoin(outputDir, f'Coreg_alphaImgCuts_{ixImg}.mhd'))
            continue

        img, transform = anatomicalCoreg(phaloidinCuts[0], phaloidin)

        phaloidinCuts[ixImg] = img
        DAPICuts[ixImg] = sitk.Transformix(DAPICuts[ixImg], transform)
        transform = correctResolutionTransform(transform, alphaImgCuts[ixImg])
        alphaImgCuts[ixImg] = sitk.Transformix(alphaImgCuts[ixImg], transform)

        sitk.WriteImage(phaloidinCuts[ixImg], pjoin(outputDir, f'Coreg_phaloidinCuts{ixImg}.mhd'))
        sitk.WriteImage(DAPICuts[ixImg], pjoin(outputDir, f'Coreg_DAPICuts_{ixImg}.mhd'))
        sitk.WriteImage(alphaImgCuts[ixImg], pjoin(outputDir, f'Coreg_alphaImgCuts_{ixImg}.mhd'))
#########################################################

# def doFullAnatomicalCoreg(PASCuts: list[sitk.Image], alphaImgCuts: list[sitk.Image], outputDir: str) -> None:

#    for ixImg, PAS in enumerate(PASCuts):
#        if ixImg == 0:
#           sitk.WriteImage(PASCuts[ixImg], pjoin(outputDir, f'Coreg_PASCuts{ixImg}.mhd'))
#           sitk.WriteImage(alphaImgCuts[ixImg], pjoin(outputDir, f'Coreg_alphaImgCuts_{ixImg}.mhd'))
#           continue

#       img, transform = anatomicalCoreg(PASCuts[0], PAS)
#       PASCuts[ixImg] = img
#       transform = correctResolutionTransform(transform, alphaImgCuts[ixImg])
#       alphaImgCuts[ixImg] = sitk.Transformix(alphaImgCuts[ixImg], transform)

#       sitk.WriteImage(PASCuts[ixImg], pjoin(outputDir, f'Coreg_PASCuts{ixImg}.mhd'))
#       sitk.WriteImage(alphaImgCuts[ixImg], pjoin(outputDir, f'Coreg_alphaImgCuts_{ixImg}.mhd'))      
######################################################################################################################################################



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


################ NZ: required changes for PAS stain ##############################################################################################

def doFullCoreg(alphaImg: sitk.Image,  dapiImage: sitk.Image, phaloidinImage: sitk.Image, matchingCoords: list[list], outputDir: str) -> sitk.Image:

    matchingCoords = matchingCoords[::-1]
    dapiImage = matchPointClouds(matchingCoords, dapiImage, alphaImg)
    phaloidinImage = matchPointClouds(matchingCoords, phaloidinImage, alphaImg)

    sitk.WriteImage(dapiImage, pjoin(
        outputDir, 'micDAPIImgPointCloudMatch.mhd'))
    sitk.WriteImage(phaloidinImage, pjoin(
        outputDir, 'micPHALImgPointCloudMatch.mhd'))


    alphaImg = sitk.Median(alphaImg, (3,3))
    dapiImageInAlpha, phaloidinInAlpha = coregAlphaCameraImgToMicImg(dapiImage, phaloidinImage, alphaImg)


    sitk.WriteImage(dapiImageInAlpha, pjoin(outputDir, '_fullDAPIImageInAlpha.mhd'))
    sitk.WriteImage(phaloidinInAlpha, pjoin(outputDir, '_fullPhaloidinImageInAlpha.mhd'))

    return dapiImageInAlpha, phaloidinInAlpha
##################################################
#def doFullCoreg(alphaImg: sitk.Image,  PASImage: sitk.Image, matchingCoords: list[list], outputDir: str) -> sitk.Image:

#    matchingCoords = matchingCoords[::-1]
#    PASImage = matchPointClouds(matchingCoords, PASImage, alphaImg)

#    sitk.WriteImage(PASImage, pjoin(outputDir, 'micPASImgPointCloudMatch.mhd'))

#    alphaImg = sitk.Median(alphaImg, (3,3))
#    PASImageInAlpha = coregAlphaCameraImgToMicImg(PASinImage, alphaImg)


#    sitk.WriteImage(PASImageInAlpha, pjoin(outputDir, '_fullDAPIImageInAlpha.mhd'))

#    return PASImageInAlpha
###############################################################################################################################################