#%%
import os
from os.path import join as pjoin
import sys

from glob import glob

import numpy as np
import SimpleITK as sitk

import matplotlib.pyplot as plt
import pickle
import copy
import matplotlib.ticker as tkr


fMainDir = r'C:\OUTPUT\iQID Coreg\September 2024'


LambdaAcDays = np.log(2)/9.9
LambdaAcSeconds = LambdaAcDays / (24*60*60)


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

def srgb2gray(image):
    # # Convert sRGB image to gray scale and rescale results to [0,255]    
    # channels = [sitk.VectorIndexSelectionCast(image,i, sitk.sitkFloat32) for i in range(image.GetNumberOfComponentsPerPixel())]
    # #linear mapping
    # I = 1/255.0*(0.2126*channels[0] + 0.7152*channels[1] + 0.0722*channels[2])
    # #nonlinear gamma correction
    # I = I*sitk.Cast(I<=0.0031308,sitk.sitkFloat32)*12.92 + I**(1/2.4)*sitk.Cast(I>0.0031308,sitk.sitkFloat32)*1.055-0.055
    # return sitk.InvertIntensity(sitk.Cast(sitk.RescaleIntensity(I), sitk.sitkUInt8))
    # image = sitk.DICOMOrient(image, "LPS")

    npImg = sitk.GetArrayFromImage(image)
    r, g, b = npImg[0,:,:], npImg[1,:,:], npImg[2,:,:]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    newImg = sitk.GetImageFromArray(gray)

    # I = 1/255.0*(0.2126*npImg[0,:,:] + 0.7152*npImg[1, :,:] + 0.0722*npImg[2, :,:])
    # # #nonlinear gamma correction
    # I = I*(I<=0.0031308)*12.92 + I**(1/2.4)*(I>0.0031308)*1.055-0.055
    # newImg = sitk.GetImageFromArray(I)

    newImg.SetOrigin(image.GetOrigin()[:2])
    newImg.SetSpacing(image.GetSpacing()[:2])

    dir = np.array(image.GetDirection()).reshape((3,-1))
    newDir = dir[:2, :2]
    newImg.SetDirection(newDir.flatten())

    ##Normalize direction to unitiy = eye(2)
    

    # newImg = sitk.InvertIntensity(newImg)
    # newImg = sitk.Normalize(newImg)

    return newImg



def getCorrespondingSections(micImages: list[sitk.Image], numSectionsPerSlide=3) -> list[list]:
    
    global coords
    numSlides = len(micImages)

    coords = [[] for _ in range(numSlides)]

    fig, ax = plt.subplots(nrows=1, ncols=numSlides)
    for numImg, micImage in enumerate(micImages):
        locImg = sitk.GetArrayFromImage(micImage).T
        mx2 = np.percentile(locImg, 99.9)
        ax[numImg].imshow(locImg, vmax = mx2, cmap='Reds')
        ax[numImg].axis('off')

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

                coords[ixx].append(micImages[ixx].TransformContinuousIndexToPhysicalPoint((ix, iy)))

                # coords[ixx].append((ix, iy))

                a.scatter(ix, iy)
                a.figure.canvas.draw()
                # fig.canvas.draw()

        # if len(coords[0]) == numSectionsPerSlide and len(coords[1]) == numSectionsPerSlide:
        isEnough = 0
        for icord in coords:
            if len(icord) == numSectionsPerSlide:
                isEnough += 1
        
        if isEnough == numSlides:

            fig.canvas.mpl_disconnect(cid)
            plt.close(fig)

            return coords

    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()

    return coords


def matchPointClouds(coords, imageStack: list[sitk.Image]) -> tuple[list[sitk.Image], list[tuple]]:
   
    # assert(len(coords) ==  len(imageStack))

    # Get corresponding points for each image
    pointsReferenceImage = np.array(coords[0])
    referenceImage = imageStack[0]
    
    # PhysicalPointsRefImg = np.array(
    #     [i * referenceImage.GetSpacing() for i in pointsReferenceImage]).T
    
    # PhysicalPointsReferenceImage = [referenceImage.TransformContinuousIndexToPhysicalPoint(i) for i in pointsReferenceImage]
    # PhysicalPointsReferenceImage = np.array(PhysicalPointsReferenceImage).T

    returnImageList = [referenceImage]
    returnTransformationList = [(referenceImage.GetOrigin(), referenceImage.GetDirection())]

    for ixImage in range(1, len(imageStack)):
        pointsMovingImg = np.array(coords[ixImage])
        movingImage = imageStack[ixImage]
        
        # PhysicalPointsMovingImg = np.array(
        #     [i * movingImage.GetSpacing() for i in pointsMovingImg]).T


        # PhysicalPointsMovingImage = np.array(
        #     [i * movingImage.GetSpacing() for i in pointsMovingImage]).T #this does not include difference in origin
        
        # PhysicalPointsMovingImage = [movingImage.TransformContinuousIndexToPhysicalPoint(i) for i in pointsMovingImg]
        # PhysicalPointsMovingImage = np.array(PhysicalPointsMovingImage).T
        # PhysicalPointsReferenceImage = np.array(
        #     [i * referenceImage.GetSpacing() for i in pointsReferenceImage]).T
        
        

        p2 = pointsMovingImg.T#PhysicalPointsMovingImage
        p1 = pointsReferenceImage.T#PhysicalPointsReferenceImage

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

        assert np.allclose(np.linalg.det(
            R), 1.0), "Rotation matrix of N-point registration not 1, see paper Arun et al."

        # Calculate translation matrix
        T = p2_c - np.matmul(R, p1_c)

        result = T + np.matmul(R, p1)

        # Calculate new direction matrix and origin
        newOrigin = [i for i in T.flatten()]
        newDirection = R.flatten()

        movingImage2 = movingImage
        movingImage2.SetDirection(newDirection)
        movingImage2.SetOrigin(newOrigin)

        returnImageList.append(movingImage2)
        returnTransformationList.append( (newOrigin, newDirection) )

    return returnImageList, returnTransformationList
####################
dirs = [pjoin(fMainDir, i) for i in os.listdir(fMainDir)]
for fDataDir in dirs:

    #%%

    numSlides = len([i for i in glob(pjoin(fDataDir, 'alphaImgHiRes_*.nii')) if '_corr.nii' not in i ])

    if numSlides < 3: continue

    alphaCameraSlides = [sitk.ReadImage(pjoin(fDataDir, f'alphaImgHiRes_{i}.nii')) for i in range(numSlides)]
    HESlides = [srgb2gray(sitk.ReadImage(pjoin(fDataDir, f'HE_{i}.nii')))  for i in range(numSlides)]


    #quick sanity check to see if all "sections" are true sections
    if len(alphaCameraSlides) > 2:
        slideSelectionSum = [np.sum(sitk.GetArrayFromImage(i)) for i in alphaCameraSlides]
        slideSelectionMask = [ i > 0.50* np.median(slideSelectionSum) for i in slideSelectionSum]
        alphaCameraSlides = [alphaCameraSlides[ix] for ix, i in enumerate(slideSelectionMask) if i == True]
        HESlides = [HESlides[ix] for ix, i in enumerate(slideSelectionMask) if i == True]
    
    if len(alphaCameraSlides) < 3: continue
    
    # _ = [print(i.GetOrigin()) for i in HESlides]
    _ = [i.SetOrigin((0.,0.)) for i in HESlides]
    _ = [i.SetDirection(np.eye(2).flatten()) for i in HESlides]
    _ = [i.SetOrigin((0.,0.)) for i in alphaCameraSlides]
    _ = [i.SetDirection(np.eye(2).flatten()) for i in alphaCameraSlides]


    #Grab matching landmarks
    if not os.path.isfile(pjoin(fDataDir,'coordsDumpDosimetry.pic')):
        coords = getCorrespondingSections(HESlides)
        with open(pjoin(fDataDir,'coordsDumpDosimetry.pic'), 'wb') as f:
            pickle.dump(coords, f)


    #%%
    #Match all the slides to the first one

    with open(pjoin(fDataDir,'coordsDumpDosimetry.pic'), 'rb') as f:
        coords = pickle.load(f)


    HESlidesLandmarkMatched, HESlidesNewOrientation = matchPointClouds(coords, HESlides)


    for ix, HESlidesMatched in enumerate( HESlidesLandmarkMatched ):
        sitk.WriteImage( HESlidesMatched, pjoin(fDataDir,f'CoregHE{ix}.nii'))

    alphaCameraLandmarkMatched, alphaCameraSlidesNewOrientation = matchPointClouds(coords, alphaCameraSlides)


    for ix, AlphaMatched in enumerate( alphaCameraLandmarkMatched ):
        sitk.WriteImage( AlphaMatched, pjoin(fDataDir,f'CoregAlpha{ix}.nii'))

    #%%

    for ix, movingImagez in enumerate(HESlidesLandmarkMatched):
        # if ix == 1: continue

        fixedImage = HESlidesLandmarkMatched[1]
        movingImage = movingImagez
        #Now coregister everything
        parMap = sitk.GetDefaultParameterMap("rigid")
        parMap['AutomaticTransformInitialization'] = ('true',)
        # parMap['Transform'] = ('SimilarityTransform',)
        # parMap['ImageSampler'] = ('RandomSparseMask',)
        parMap['MaximumNumberOfIterations'] = ("2580",)
        parMap['NumberOfResolutions'] = ('4',)
        # parMap['Metric'] = ('AdvancedNormalizedCorrelation',)

        parMap2 = sitk.GetDefaultParameterMap("rigid")
        parMap2['Transform'] = ('SimilarityTransform',)
        # parMap2['Metric'] = ('AdvancedNormalizedCorrelation',)
        parMap2['AutomaticTransformInitialization'] = ('false',)
        # parMap2['ImageSampler'] = ('RandomSparseMask',)
        parMap2['NumberOfResolutions'] = ('1',)
        parMap2['NumberOfSpatialSamples'] = (
            str(int(np.prod(movingImage.GetSize()) * 0.10)),)
        parMap2['MaximumNumberOfIterations'] = ("4000",)
        # parMap2['ASGDParameterEstimationMethod'] = ("DisplacementDistribution",)
        # parMap2['NumberOfSpatialSamples'] = (
        #     str(int(np.prod(fixedImage.GetSize()) * 0.10)),)
        # parMap2['MaximumNumberOfIterations'] = ("4000",)
        # parMap2['CheckNumberOfSamples'] = ('false',)

        parMap3 = sitk.GetDefaultParameterMap('bspline')
        # parMap3['NumberOfResolutions']= ("1",)
        # parMap3['GridSpacingSchedule'] = (str(1.4 * alphaImg.GetSpacing()[0]), str(1.0 * alphaImg.GetSpacing()[0]))
        # parMap3['GridSpacingSchedule'] = ("5",)
        parMap3['ASGDParameterEstimationMethod'] = ("DisplacementDistribution",)
        parMap3['FinalGridSpacingInPhysicalUnits'] = (str(15*movingImage.GetSpacing()[0]),) #changed from 25 to see if we can fix small elastic deformation differences
        parMap3['GridSpacingSchedule'] = (str(10.0 * movingImage.GetSpacing()[0]),str(5.0 * movingImage.GetSpacing()[0]),str(2.0 * movingImage.GetSpacing()[0]),str(1.0 * movingImage.GetSpacing()[0]),)
        parMap3['ASGDParameterEstimationMethod'] = ("DisplacementDistribution",)


        # parMap3['MovingImagePyramid'] = ("MovingShrinkingImagePyramid",)
        # parMap3['FixedImagePyramid'] = ("FixedShrinkingImagePyramid",)
        parMap3['Metric'] = ("AdvancedMattesMutualInformation","TransformBendingEnergyPenalty")
        # parMap3['Metric'] = ("AdvancedNormalizedCorrelation","TransformBendingEnergyPenalty")
        parMap3['Metric1Weight'] = ("2.5",)
        parMap3['MaximumNumberOfIterations'] = ("2000",)
        #Important to do this or not? Might want to keep scaling??


        parMaps = sitk.VectorOfParameterMap()
        parMaps.append(parMap)
        parMaps.append(parMap2)
        parMaps.append(parMap3)

        elastixImageFilter = sitk.ElastixImageFilter()
        elastixImageFilter.SetParameterMap(parMaps)
        elastixImageFilter.SetMovingImage(movingImagez)
        elastixImageFilter.SetFixedImage(fixedImage)
        elastixImageFilter.SetLogToFile(False)
        elastixImageFilter.SetLogToConsole(True)
        elastixImageFilter.Execute()

        transMap = elastixImageFilter.GetTransformParameterMap()

        sitk.WriteImage(elastixImageFilter.GetResultImage(), pjoin(fDataDir, f'movingImg_{ix}.nii'))

        ##Apply the same transmap to the alpha camera images

        numTransforms = len(transMap)
        transforms = [sitk.ReadParameterFile(f'TransformParameters.{i}.txt') for i in range(numTransforms)]
        #making sure initial tansfroms are correct
        for trans in transforms:
            trans['InitialTransformParametersFileName'] =  ("NoInitialTransform",)

        transforms2 = correctResolutionTransform(transforms, alphaCameraLandmarkMatched[ix])
    
        strx = sitk.TransformixImageFilter()
        strx.ComputeDeterminantOfSpatialJacobianOn()
        strx.SetTransformParameterMap(transforms2)
        strx.SetMovingImage(alphaCameraLandmarkMatched[ix])
        strx.SetOutputDirectory('.')
        strx.Execute()
        correctedAlphaCamera = strx.GetResultImage()
        jacobian = sitk.ReadImage('spatialJacobian.nii')
        jacobian = sitk.Resample(jacobian, correctedAlphaCamera, sitk.Transform(), sitk.sitkNearestNeighbor)
        correctedAlphaCamera = correctedAlphaCamera * jacobian

        # correctedAlphaCamera = sitk.Transformix( alphaCameraLandmarkMatched[ix], transforms )
        sitk.WriteImage(correctedAlphaCamera, pjoin(fDataDir,f'movingAlphaCamera_{ix}.nii'))



    # sitk.WriteImage( alphaCameraSlides[1], pjoin(fDataDir,f'movingAlphaCamera_1.nii'))

    # sitk.WriteImage(HESlides[1], pjoin(fDataDir,'movingImg_1.nii'))

    #%% Now Run Dose Rate Calc For middle slide

    fAlphaCameraImages = [pjoin(fDataDir, i) for i in glob('movingAlphaCamera_*.nii', root_dir=fDataDir)]
    fAlphaCameraImages = 5* [fAlphaCameraImages[0]] + [fAlphaCameraImages[1]] + 5*[fAlphaCameraImages[2]]
    alphaCameraImages = sitk.JoinSeries( [sitk.ReadImage(i, sitk.sitkFloat32) for i in fAlphaCameraImages] )

    alphaCameraImages.SetSpacing( list(alphaCameraImages.GetSpacing()[:2]) + [14.*2] )
    alphaCameraImages.SetOrigin( list(alphaCameraImages.GetOrigin()[:2]) + [-5*14.] )


    fHEIntensityImages = [pjoin(fDataDir, i) for i in glob('movingImg_*.nii', root_dir=fDataDir)]
    fHEIntensityImages = 5* [fHEIntensityImages[0]] + [fHEIntensityImages[1]] + 5*[fHEIntensityImages[2]]
    HEIntensitySlides = sitk.JoinSeries( [sitk.ReadImage(i) for i in fHEIntensityImages] )


    HEIntensitySlides.SetSpacing( list(HEIntensitySlides.GetSpacing()[:2]) + [14.*2] )
    HEIntensitySlides.SetOrigin( list(HEIntensitySlides.GetOrigin()[:2]) + [-5.*14.] )


    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputOrigin(alphaCameraImages.GetOrigin())
    resampler.SetOutputSpacing( list(alphaCameraImages.GetSpacing()[:2]) + [14.] )
    resampler.SetOutputDirection(alphaCameraImages.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)
    sz = list(alphaCameraImages.GetSize())
    sz[-1] = sz[-1]*2 - 1
    resampler.SetSize( sz )
    interPolatedAlphaCameraImages = resampler.Execute(alphaCameraImages)

    sitk.WriteImage(interPolatedAlphaCameraImages, pjoin(fDataDir, 'interpolatedAlphaCameraImages.nii'))

    #Do a simple DVK
    try:
        DVK = sitk.ReadImage(r'DVK_DosePerDecay_Ac225_26_26_14.nii', sitk.sitkFloat32)
    except:
        DVK = sitk.ReadImage(r'../DVK_DosePerDecay_Ac225_26_26_14.nii', sitk.sitkFloat32)

    DVK = sitk.PermuteAxes(DVK, [2,1,0])
    doseRateMap = sitk.Convolution( interPolatedAlphaCameraImages, DVK, normalize=False )

    npDoseRateMap = sitk.GetArrayFromImage(doseRateMap)
    centralSliceNum = int(npDoseRateMap.shape[0]/2)
    npCentralSlice = npDoseRateMap[centralSliceNum,:: ]

    centralSlice = sitk.GetImageFromArray(npCentralSlice)
    centralSlice.SetSpacing(doseRateMap.GetSpacing())
    sitk.WriteImage( centralSlice, pjoin(fDataDir, 'DoseRateMap.nii'))

    npHEIntensitySlides = sitk.GetArrayFromImage(HEIntensitySlides)
    centralSliceNum =  int(npHEIntensitySlides.shape[0]/2)
    centralHESlice = sitk.GetImageFromArray(npHEIntensitySlides[centralSliceNum,::])
    centralHESlice.SetSpacing(HEIntensitySlides.GetSpacing())
    sitk.WriteImage( centralHESlice, pjoin(fDataDir, 'CentralHEIntensityMap.nii'))


    #Can we overlay this with our current HE section?

    npCentralHESlide = sitk.GetArrayFromImage(centralHESlice)
    npHE = npCentralHESlide

    alphaImg = sitk.Resample(centralSlice, centralHESlice, sitk.Transform(), sitk.sitkLinear)
    # sitk.WriteImage(alphaImg, pjoin(outputDir, f'alphaImgOverlay_{ix}.nii'))

    # Plot with contours [cut outs]
    npAlphaImg = sitk.GetArrayFromImage(alphaImg)
    npAlphaImg[npAlphaImg <=0] = 0

    npAlphaImg /= npAlphaImg.sum()
    npAlphaImg *= npCentralSlice.sum()

    #Activity to total number of decays
    # npAlphaImg /= LambdaAcSeconds

    npAlphaImg *= 1E3 #Gy/s to milliGy/s
    npAlphaImg *= (60*60) #cGy/h
    npHE[npHE < 1E-4] = 1 #small fix from resampling and image intensity inversion

    fig, ax = plt.subplots()
    fig.set_size_inches((10, 10))
    plt.axis("off")
    im = ax.imshow(npHE, cmap = 'Grays')
    # levels = np.linspace(npAlphaImg.min(), np.percentile(npAlphaImg, 99.999), 5)
    levels = np.linspace(np.percentile(npAlphaImg, 35), npAlphaImg.max(), 10)
    CS = ax.contourf(npAlphaImg, levels, linewidths=0.1, alpha = 0.2, cmap='jet')

    CB = fig.colorbar(CS, shrink=0.4, format=tkr.FormatStrFormatter('%.2g'), pad=0.03)
    l, b, w, h = ax.get_position().bounds
    ll, bb, ww, hh = CB.ax.get_position().bounds
    CB.ax.set_position([ll, b + 0.1 * h, ww, h * 0.8])
    CB.set_label('Dose rate [mGy/hour]', rotation=270, labelpad=15 )
    plt.tight_layout()
    plt.savefig(pjoin(fDataDir, f"DoseRate_Contours.tif"), dpi=600)


    # fig, ax = plt.subplots()
    # fig.set_size_inches((10, 10))
    # plt.axis("off")
    # im = ax.imshow(npHE, cmap = 'Grays')
    # plt.savefig(pjoin(fDataDir, f"HEGreys.tif"), dpi=600)
    # %%
