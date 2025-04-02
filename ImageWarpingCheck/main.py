#%%
import SimpleITK as sitk
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
sys.path.append('../__GenPlotsForPaper') #bad...
sys.path.append('__GenPlotsForPaper') #bad...
from plotOverlay import overlay_plot
import matplotlib.widgets as widgets
from mpl_toolkits.axes_grid1 import make_axes_locatable

DOTDIAM = 600 #micrometer
LINEWIDTH = 300 #Micrometer

DOTSPACING= 1000 #micrometer
FOV = 100000 #micrometer
PIXELSPACING = 1/18.85 * 1000  #* 0.998701 #micron/pix

# fDotPattern = r'C:\Users\remco\Washington University in St. Louis\Bastiaannet Lab - Lab Management - Lab Management\Materials and Equipment\iQID\Johns Hopkins - iQID Mega -Scale and Dots\_DotsSUM_CameraID-21249222_images.tif'
# fSquarePattern = r'C:\Users\remco\Washington University in St. Louis\Bastiaannet Lab - Lab Management - Lab Management\Materials and Equipment\iQID\Johns Hopkins - iQID Mega -Scale and Dots\xy_SUM_CameraID-21249222_images.tif'
fDotPattern = 'Dots.nii'
fSquarePattern = r'Squares.nii'

dotSpacingPixels = DOTSPACING/PIXELSPACING

def readAndFilterPattern(fFile):

    dotPatternMeasured = sitk.ReadImage(fFile)
    dotPatternMeasured.SetSpacing([PIXELSPACING]*2)
    dotPatternMeasuredFilt = sitk.DiscreteGaussian(dotPatternMeasured, 50., 1000, 0.01, False)

    dotPatternMeasured = dotPatternMeasured - dotPatternMeasuredFilt
    dotPatternMeasured = sitk.Normalize(dotPatternMeasured)
    return dotPatternMeasured

def generateDotsPattern(dotPatternMeasured: sitk.Image, maxSize : int|None = None):
    
    npDotsGen = np.zeros_like(sitk.GetArrayViewFromImage(dotPatternMeasured)).astype(np.uint8)

    numDotsXY = [i / dotSpacingPixels for i in npDotsGen.shape]

    xLocs = np.linspace(0, npDotsGen.shape[0]-1, int(np.ceil(numDotsXY[0])))
    yLocs = np.linspace(0, npDotsGen.shape[1]-1, int(np.ceil(numDotsXY[1])))

    if maxSize is not None:

        #clean everything further away than maxSize x xLocs
        midpoint = [int(np.round(i/2.)) for i in npDotsGen.shape]

        xLocs = xLocs[np.abs(xLocs - midpoint[0]) < (maxSize * dotSpacingPixels)]
        yLocs = yLocs[np.abs(yLocs - midpoint[1]) < (maxSize * dotSpacingPixels)]

    for iX in xLocs:
        for iY in yLocs:
            npDotsGen[int(iX), int(iY)] = 1


    # npDotsGen[:, 0:230] = 0
    # npDotsGen[:, 2200:] = 0
    # npDotsGen[:170,:] = 0


    dotsGen = sitk.GetImageFromArray(npDotsGen.astype(np.uint8))
    dotsGen.CopyInformation(dotPatternMeasured)

    distMap = sitk.SignedMaurerDistanceMap(dotsGen, useImageSpacing=True, squaredDistance=False)
    dotsGen = (distMap > (DOTDIAM/2.)) #Because the actual measured map is also inverted

    dotsGen = sitk.Normalize(dotsGen)
    return dotsGen

def generateSquaresPattern(dotPatternMeasured: sitk.Image, maxSize = 100):

    # Generate Square pattern
    npSquareGen = np.zeros_like(sitk.GetArrayViewFromImage(dotPatternMeasured)).astype(np.uint8)
    midPoint = [int( i /2. ) for i in npSquareGen.shape]

    squareWidth = 4.9 #mm
    while squareWidth <= maxSize:
        squareHalfWidthPixels = int(np.round((squareWidth*1E3) / PIXELSPACING /2))
        xes = np.arange(midPoint[0] - squareHalfWidthPixels, midPoint[0] + squareHalfWidthPixels)
        yes = np.arange(midPoint[1] - squareHalfWidthPixels, midPoint[1] + squareHalfWidthPixels)

        #Gen Horizontal lines
        maxY = midPoint[1] + squareHalfWidthPixels
        minY = midPoint[1] - squareHalfWidthPixels
        npSquareGen[xes, maxY] = 1
        npSquareGen[xes, minY] = 1

        #Gen Vertical Lines
        maxX = midPoint[0] + squareHalfWidthPixels
        minX = midPoint[0] - squareHalfWidthPixels
        npSquareGen[maxX, yes] = 1
        npSquareGen[minX, yes] = 1

        squareWidth += 4.9

    squareGen = sitk.GetImageFromArray(npSquareGen.astype(np.uint8))
    squareGen.CopyInformation(squarePatternMeasured)

    distMap = sitk.SignedMaurerDistanceMap(squareGen, useImageSpacing=True, squaredDistance=False)
    squareGen = (distMap >= (LINEWIDTH/2.)) #Because the actual measured map is also inverted
    squareGen = sitk.Normalize(squareGen)

    return squareGen

def generateWarpingGratePattern(dotPatternMeasured: sitk.Image, maxSize = 100):

    # Generate Square pattern
    npSquareGen = np.zeros_like(sitk.GetArrayViewFromImage(dotPatternMeasured)).astype(np.uint8)
    # midPoint = [int( i /2. ) for i in npSquareGen.shape]

    # squareWidth = 5 #mm
    lineSpacingPx = int(dotSpacingPixels) *3
    numVertLines = int(npSquareGen.shape[0] / lineSpacingPx)
    numHorLines =  int(npSquareGen.shape[1] / lineSpacingPx)

    #Do vertical lines
    for iVert in range(numVertLines+1):
        npSquareGen[iVert * lineSpacingPx, :] = 1
    
    
    for iHorz in range(numHorLines+1):
        npSquareGen[:, iHorz * lineSpacingPx] = 1

    squareGen = sitk.GetImageFromArray(npSquareGen.astype(np.uint8))
    squareGen.CopyInformation(dotPatternMeasured)

    distMap = sitk.SignedMaurerDistanceMap(squareGen, useImageSpacing=True, squaredDistance=False)
    squareGen = (distMap >= (LINEWIDTH/2.)) #Because the actual measured map is also inverted
    squareGen = sitk.Normalize(squareGen)

    return squareGen



def getModelToMeasuredInitialTransform(dotsGen: sitk.Image, dotsPatternMeasured: sitk.Image):
    parMap2 = sitk.GetDefaultParameterMap("rigid")
    parMap2['Transform'] = ('SimilarityTransform',)
    parMap2['AutomaticTransformInitialization'] = ('false',)
    # parMap2['ImageSampler'] = ('RandomSparseMask',)
    parMap2['NumberOfResolutions'] = ('4',)
    # parMap2['ASGDParameterEstimationMethod'] = ("DisplacementDistribution",)
    parMap2['MaximumNumberOfIterations'] = ("1000",)
    parMap2['Metric'] = ('AdvancedNormalizedCorrelation',)
    # parMap2['Optimizer'] = ( 'FullSearch',)
    # parMap2["AutomaticTransformInitializationMethod"] = ("CenterOfGravity",) 
    # parMap2['CheckNumberOfSamples'] = ('false',)
    



    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetParameterMap(parMap2)
    elastixImageFilter.SetMovingImage(dotsGen)

    elastixImageFilter.SetFixedImage(dotsPatternMeasured)
    elastixImageFilter.SetLogToFile(False)
    elastixImageFilter.SetLogToConsole(True)
    elastixImageFilter.Execute()

    dotsGenInMeasured = elastixImageFilter.GetResultImage()
    # trans = elastixImageFilter.GetTransformParameterMap()

    transform = sitk.ReadParameterFile(f'TransformParameters.{0}.txt') 
    transform['InitialTransformParametersFileName'] =  ("NoInitialTransform",)

    print(transform.asdict())
    return dotsGenInMeasured, transform

def  getPerfectModelToMeasurement(dotsPatternMeasured: sitk.Image, dotsGen: sitk.Image, dotsGenSmall: sitk.Image|None = None):

    if dotsGenSmall is None:        
        dotsGenInMeasured, initialTrans = getModelToMeasuredInitialTransform(dotsGen, dotsPatternMeasured)
    else:
        _, initialTrans = getModelToMeasuredInitialTransform(dotsGenSmall, dotsPatternMeasured)
        dotsGenInMeasured = sitk.Transformix(dotsGen, initialTrans)
    
    return dotsGenInMeasured, initialTrans

def getCorrectionTransform( dotsPatternMeasured: sitk.Image, dotsGenInMeasured: sitk.Image):



    parMap3 = sitk.GetDefaultParameterMap('bspline')
    parMap3['NumberOfResolutions']= ("1",)
    # parMap3['GridSpacingSchedule'] = (str(1.4 * alphaImg.GetSpacing()[0]), str(1.0 * alphaImg.GetSpacing()[0]))
    parMap3['GridSpacingSchedule'] = ("1.0",)
    parMap3['ASGDParameterEstimationMethod'] = ("DisplacementDistribution",)
    parMap3['FinalGridSpacingInPhysicalUnits'] = (str(1000*dotsPatternMeasured.GetSpacing()[0]),)
    parMap3['MovingImagePyramid'] = ("MovingShrinkingImagePyramid",)
    parMap3['FixedImagePyramid'] = ("FixedShrinkingImagePyramid",)
    # parMap3['Metric'] = ("AdvancedMattesMutualInformation","TransformBendingEnergyPenalty")
    parMap3['Metric'] = ("AdvancedNormalizedCorrelation","TransformBendingEnergyPenalty")
    # parMap3['Metric1Weight'] = ("1.5",)
    parMap3['Metric1Weight'] = ("100.5",)
    parMap3['MaximumNumberOfIterations'] = ("5000",)
    # parMap3['MaximumStepLength'] = ("1.2",)
    parMap3['MaximumDisplacement'] = ("500",)


    parMap4 = sitk.GetDefaultParameterMap('bspline')
    parMap4['NumberOfResolutions']= ("1",)
    # parM4p3['GridSpacingSchedule'] = (str(1.4 * alphaImg.GetSpacing()[0]), str(1.0 * alphaImg.GetSpacing()[0]))
    parMap4['GridSpacingSchedule'] = ("1.0",)
    parMap4['ASGDParameterEstimationMethod'] = ("DisplacementDistribution",)
    parMap4['FinalGridSpacingInPhysicalUnits'] = (str(200*dotsPatternMeasured.GetSpacing()[0]),)
    parMap4['MovingImagePyramid'] = ("MovingShrinkingImagePyramid",)
    parMap4['FixedImagePyramid'] = ("FixedShrinkingImagePyramid",)
    # parMap4['Metric'] = ("AdvancedMattesMutualInformation","TransformBendingEnergyPenalty")
    parMap4['Metric'] = ("AdvancedNormalizedCorrelation","TransformBendingEnergyPenalty")
    # parMap4['Metric1Weight'] = ("1.5",)
    parMap4['Metric1Weight'] = ("10.5",)
    parMap4['MaximumNumberOfIterations'] = ("5000",)
    parMap4['MaximumDisplacement'] = ("500",)
    # parMap4['MaximumStepLength'] = ("1.2",)

    parMaps = sitk.VectorOfParameterMap()
    parMaps.append(parMap3)
    parMaps.append(parMap4)


    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetParameterMap(parMaps)

    elastixImageFilter.SetMovingImage(dotsPatternMeasured)
    elastixImageFilter.SetFixedImage(dotsGenInMeasured)
    elastixImageFilter.SetLogToFile(False)
    elastixImageFilter.SetLogToConsole(True)
    elastixImageFilter.Execute()

    CorrectedMeasured = elastixImageFilter.GetResultImage()

    TransMeasurementToPerfect = elastixImageFilter.GetTransformParameterMap()
    for trans in TransMeasurementToPerfect:
        trans['InitialTransformParametersFileName'] = ('NoInitialTransform',)

    return CorrectedMeasured, TransMeasurementToPerfect


def getCorrectionTransformAffine( dotsPatternMeasured: sitk.Image, dotsGenInMeasured: sitk.Image):

    # dotsGenInMeasured = sitk.InvertIntensity(dotsGenInMeasured)
    # dotsPatternMeasured = sitk.InvertIntensity(dotsPatternMeasured)

    parMap2 = sitk.GetDefaultParameterMap('rigid')
    parMap2['NumberOfResolutions']= ("4",)
    # parMap3['GridSpacingSchedule'] = (str(1.4 * alphaImg.GetSpacing()[0]), str(1.0 * alphaImg.GetSpacing()[0]))
    # parMap3['GridSpacingSchedule'] = ("1.0",)
    # parMap3['ASGDParameterEstimationMethod'] = ("DisplacementDistribution",)
    # parMap3['FinalGridSpacingInPhysicalUnits'] = (str(1000*dotsPatternMeasured.GetSpacing()[0]),)
    # parMap3['MovingImagePyramid'] = ("MovingShrinkingImagePyramid",)
    # parMap3['FixedImagePyramid'] = ("FixedShrinkingImagePyramid",)
    # parMap3['Metric'] = ("AdvancedMattesMutualInformation","TransformBendingEnergyPenalty")
    # parMap3['Metric'] = ("AdvancedNormalizedCorrelation","TransformBendingEnergyPenalty")
    # parMap3['Metric1Weight'] = ("1.5",)
    # parMap3['Metric1Weight'] = ("100.5",)
    parMap2['MaximumNumberOfIterations'] = ("5000",)
    # parMap3['MaximumStepLength'] = ("1.2",)

    parMap3 = sitk.GetDefaultParameterMap('affine')
    parMap3['NumberOfResolutions']= ("4",)
    # parMap3['GridSpacingSchedule'] = (str(1.4 * alphaImg.GetSpacing()[0]), str(1.0 * alphaImg.GetSpacing()[0]))
    # parMap3['GridSpacingSchedule'] = ("1.0",)
    # parMap3['ASGDParameterEstimationMethod'] = ("DisplacementDistribution",)
    # parMap3['FinalGridSpacingInPhysicalUnits'] = (str(1000*dotsPatternMeasured.GetSpacing()[0]),)
    # parMap3['MovingImagePyramid'] = ("MovingShrinkingImagePyramid",)
    # parMap3['FixedImagePyramid'] = ("FixedShrinkingImagePyramid",)
    # parMap3['Metric'] = ("AdvancedMattesMutualInformation","TransformBendingEnergyPenalty")
    # parMap3['Metric'] = ("AdvancedNormalizedCorrelation","TransformBendingEnergyPenalty")
    # parMap3['Metric1Weight'] = ("1.5",)
    # parMap3['Metric1Weight'] = ("100.5",)
    parMap3['MaximumNumberOfIterations'] = ("5000",)
    # parMap3['MaximumStepLength'] = ("1.2",)


    parMap4 = sitk.GetDefaultParameterMap('bspline')
    parMap4['NumberOfResolutions']= ("1",)
    # parM4p3['GridSpacingSchedule'] = (str(1.4 * alphaImg.GetSpacing()[0]), str(1.0 * alphaImg.GetSpacing()[0]))
    parMap4['GridSpacingSchedule'] = ("1.0",)
    parMap4['ASGDParameterEstimationMethod'] = ("DisplacementDistribution",)
    parMap4['FinalGridSpacingInPhysicalUnits'] = (str(200*dotsPatternMeasured.GetSpacing()[0]),)
    parMap4['MovingImagePyramid'] = ("MovingShrinkingImagePyramid",)
    parMap4['FixedImagePyramid'] = ("FixedShrinkingImagePyramid",)
    # parMap4['Metric'] = ("AdvancedMattesMutualInformation","TransformBendingEnergyPenalty")
    parMap4['Metric'] = ("AdvancedNormalizedCorrelation","TransformBendingEnergyPenalty")
    # parMap4['Metric1Weight'] = ("1.5",)
    parMap4['Metric1Weight'] = ("10.5",)
    parMap4['MaximumNumberOfIterations'] = ("5000",)
    # parMap4['MaximumStepLength'] = ("1.2",)


    parMaps = sitk.VectorOfParameterMap()
    parMaps.append(parMap2)
    parMaps.append(parMap3)
    parMaps.append(parMap4)

    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetParameterMap(parMaps)

    elastixImageFilter.SetMovingImage(dotsPatternMeasured)
    elastixImageFilter.SetFixedImage(dotsGenInMeasured)
    elastixImageFilter.SetLogToFile(False)
    elastixImageFilter.SetLogToConsole(True)
    elastixImageFilter.Execute()

    CorrectedMeasured = elastixImageFilter.GetResultImage()

    TransMeasurementToPerfect = elastixImageFilter.GetTransformParameterMap()
    for trans in TransMeasurementToPerfect:
        trans['InitialTransformParametersFileName'] = ('NoInitialTransform',)

    return CorrectedMeasured, TransMeasurementToPerfect

def overlay_plot(image1, image2, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(sitk.GetArrayFromImage(image1), cmap='gray', alpha=0.5)
    ax.imshow(sitk.GetArrayFromImage(image2), cmap='jet', alpha=0.5)
    return ax

#%%

dotPatternMeasured = readAndFilterPattern(fDotPattern)

squarePatternMeasured = readAndFilterPattern(fSquarePattern)

#%% Select relevant region only
def crop_image(image, crop_coords):
    x_min, x_max, y_min, y_max = crop_coords
    cropped_array = sitk.GetArrayFromImage(image)[y_min:y_max, x_min:x_max]
    cropped_image = sitk.GetImageFromArray(cropped_array)
    cropped_image.SetSpacing(image.GetSpacing())
    cropped_image.SetOrigin(image.GetOrigin())
    cropped_image.SetDirection(image.GetDirection())
    return cropped_image

def onselect(eclick, erelease):
    global crop_coords
    x_min, x_max = int(eclick.xdata), int(erelease.xdata)
    y_min, y_max = int(eclick.ydata), int(erelease.ydata)
    crop_coords = (x_min, x_max, y_min, y_max)
    plt.close()

def select_crop_area(image):
    global crop_coords
    crop_coords = None
    fig, ax = plt.subplots()
    ax.imshow(sitk.GetArrayFromImage(image), cmap='gray')
    rs = widgets.RectangleSelector(ax, onselect, interactive=True, useblit=True,
                                   button=[1], minspanx=5, minspany=5, spancoords='pixels')
    plt.show(block=True)  # Ensure the plot is blocking
    return crop_coords

# Plot squarePatternMeasured and select crop area
if os.path.exists('crop_coords.npy'):
    crop_coords = np.load('crop_coords.npy', allow_pickle=True)
else:
    crop_coords = select_crop_area(squarePatternMeasured)
    np.save('crop_coords.npy', crop_coords)

dotPatternMeasured = crop_image(dotPatternMeasured, crop_coords)
squarePatternMeasured = crop_image(squarePatternMeasured, crop_coords)

#%%
dotsGenSmall = generateDotsPattern(dotPatternMeasured, maxSize = 5)
dotsGenSmall.SetOrigin((+1250, -1420))
sitk.WriteImage(dotsGenSmall, 'dotsSmallPatternGenerated.nii')

dotsGen = generateDotsPattern(dotPatternMeasured)
dotsGenSmall.SetOrigin((+1250, -1420))
sitk.WriteImage(dotsGen, 'dotsPatternGenerated.nii')

squaresGenSmall = generateSquaresPattern(squarePatternMeasured, maxSize=60)
squaresGenSmall.SetOrigin((+1250, -1420))
sitk.WriteImage(squaresGenSmall, 'squaresGenSmall.nii')

squaresGen = generateSquaresPattern(squarePatternMeasured)
squaresGen.SetOrigin((+1250, -1420))

sitk.WriteImage(squaresGen, 'squaresPatternGenerated.nii')


#####
##################
### Coregister ###
##################

squaresGenInMeasured, initialTrans = getPerfectModelToMeasurement(squarePatternMeasured, squaresGen, squaresGenSmall)
squaresFixed, squaresMeasuredToperfect = getCorrectionTransformAffine(squarePatternMeasured,squaresGenInMeasured)

sitk.WriteImage(squarePatternMeasured, 'squarePatternMeasured.nii')
sitk.WriteImage(squaresFixed, 'squaresFixed.nii')
sitk.WriteImage(squaresGenInMeasured, 'squaresGenInMeasured.nii')


#%%
# FIX THIS PART!!!!

dotsGenInMeasured, initialTrans = getPerfectModelToMeasurement(dotPatternMeasured, dotsGen, dotsGenSmall)
# dotsSemiFixed = sitk.Transformix(dotPatternMeasured, squaresMeasuredToperfect)
# CorrectedDotsMeasured, TransMeasurementToPerfect = getCorrectionTransform(dotsSemiFixed, dotsGenInMeasured)
CorrectedDotsMeasured = sitk.Transformix(dotPatternMeasured, squaresMeasuredToperfect)

sitk.WriteImage(dotsGenInMeasured, 'dotGenInMeasured.nii')
sitk.WriteImage(CorrectedDotsMeasured, 'CorrectedDotsMeasured.nii')
sitk.WriteImage(dotPatternMeasured, 'dotPatternMeasured.nii')


fig = overlay_plot(dotPatternMeasured, CorrectedDotsMeasured)
plt.gcf().savefig('CorrectedDotsOverlay.png', dpi = 900)




#%%###########################################
## Generate Square grid to show deformation ##
##############################################

squareWarping = generateWarpingGratePattern(dotPatternMeasured)
# squareWarping.SetOrigin((+1250, -1420))
squaredWarpingWarped = sitk.Transformix( squareWarping, squaresMeasuredToperfect )

# #Get Inverse Transform
# staightenedSquaredWarp, TransInverted = getCorrectionTransform(squaredWarpingWarped, squareWarping)
# corruptedSquaredWarp = sitk.Transformix(squareWarping, TransInverted)

transFilt = sitk.TransformixImageFilter()
transFilt.SetTransformParameterMap(squaresMeasuredToperfect)
transFilt.SetMovingImage(squareWarping)
transFilt.SetComputeDeformationField(True)
transFilt.SetComputeSpatialJacobian(True)
transFilt.SetLogToFile(True)
_ = transFilt.Execute()

defField = transFilt.GetDeformationField()
displacement = sitk.VectorMagnitude(defField)
sitk.WriteImage(displacement, 'deformationField.nii')

#create nice looking plot with deformation field magnitude
plt.figure()
plt.imshow(sitk.GetArrayFromImage(displacement), cmap = 'hot')
plt.colorbar()
plt.axis('off')
plt.savefig('DeformationField.png', dpi = 900)


#plot squarepatternmeasured and make gui to select square to crop image. Apply this crop to the corrected image and deformation field
#%%


# if crop_coords:
#     # Apply crop to the corrected image and deformation field
#     cropped_squarePatternMeasured = crop_image(squarePatternMeasured, crop_coords)
#     cropped_CorrectedDotsMeasured = crop_image(CorrectedDotsMeasured, crop_coords)
#     cropped_displacement = crop_image(displacement, crop_coords)
#     cropped_corruptedSquaredWarp = crop_image(corruptedSquaredWarp, crop_coords)
#     cropped_fixedSquareWarp = crop_image(fixedSquareWarp, crop_coords)

#     # Save cropped images
#     sitk.WriteImage(cropped_squarePatternMeasured, 'cropped_squarePatternMeasured.nii')
#     sitk.WriteImage(cropped_CorrectedDotsMeasured, 'cropped_CorrectedDotsMeasured.nii')
#     sitk.WriteImage(cropped_displacement, 'cropped_deformationField.nii')
#     sitk.WriteImage(cropped_corruptedSquaredWarp, 'cropped_corruptedSquaredWarp.nii')
#     sitk.WriteImage(cropped_fixedSquareWarp, 'cropped_fixedSquareWarp.nii')
#%%
    # Generate overlay plot for cropped images
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'wspace': 0.0})

    # Panel A: Overlay plot
    overlay_plot(squareWarping, squaredWarpingWarped, ax=ax1)
    # ax1.set_title('Panel A: Overlay Plot')
    ax1.axis('off')  # Remove axes

    divider1 = make_axes_locatable(ax1)
    a = divider1.append_axes("right", size="5%", pad=0.05)
    a.axis('off')

    # Panel B: Cropped displacement image with color bar
    im = ax2.imshow(sitk.GetArrayFromImage(displacement), cmap='hot')
    # ax2.set_title('Panel B: Displacement Field')
    ax2.axis('off')  # Remove axes

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Magnitude displacement field [$\mu$m]')  # Add units to color bar

    # Add 1 cm spacing bar to Panel A
    linelength = 10 / (PIXELSPACING * 1E-3)  # MM/px
    x_start = ax1.get_xlim()[1] - linelength - 100
    x_end = ax1.get_xlim()[1] - 100
    y_pos = ax1.get_ylim()[0] - 50
    ax1.plot([x_start, x_end], [y_pos, y_pos], linewidth=3, color='w')
    ax1.text((x_start + x_end) / 2, y_pos - 20, '1 cm', color='w', ha='center')

    fig.tight_layout(pad=0)
    plt.show()
    # fig.savefig('Cropped_ExtendOfWarpingAlphaCamera.png', dpi=600)

fixedSquareWarp = sitk.Transformix( corruptedSquaredWarp, TransMeasurementToPerfect )

#%%
fig = overlay_plot( corruptedSquaredWarp, fixedSquareWarp )
# plt.imshow(sitk.GetArrayFromImage(dotPatternMeasured), alpha = .1)

linelength = 10/(PIXELSPACING * 1E-3) #MM/px

fig.text(.803, .1, '1 cm', fontdict={'fontsize': 11, 'color': 'w'})
plt.plot([2100, 2100+linelength], [1930, 1930], linewidth = 3, color = 'w')
fig.savefig('ExtendOfWarpingAlphaCamera.png', dpi = 600)

#%%
## Do we have an equal number of dots in both cases?
dotsCorrTH = sitk.OtsuThreshold(CorrectedDotsMeasured, 1, 0)
cc = sitk.ConnectedComponent(dotsCorrTH)
cc = sitk.RelabelComponent(cc, 30)
numDotsCorrected = len(np.unique(sitk.GetArrayViewFromImage(cc)))

dotsMeasuredTH = sitk.OtsuThreshold(dotPatternMeasured, 1, 0)
cc = sitk.ConnectedComponent(dotsMeasuredTH)
cc = sitk.RelabelComponent(cc, 30)
numDotsMeasured = len(np.unique(sitk.GetArrayViewFromImage(cc)))

print(f"numDots In measured: {numDotsMeasured}")
print(f"numDots In Corrected: {numDotsCorrected}")
#YES!!!

#%%

# CorrectedMeasured, TransMeasurementToPerfect, squaresInMeasured = getCorrectionTransform(squarePatternMeasured, squaresGenInMeasured)

# sitk.WriteImage(CorrectedMeasured, 'correctedSquresMeasured.nii')
# sitk.WriteImage(squarePatternMeasured, 'squarePatternMeasured.nii')
# sitk.WriteImage(squaresInMeasured, 'squaresInMeasured.nii')

# dotsGenInMeasuredFixed, initialTrans = getPerfectModelToMeasurement(dotPatternMeasured, dotsGen, dotsGenSmall)
# CorrectedMeasuredDots = sitk.Transformix(dotPatternMeasured, TransMeasurementToPerfect)

# CorrectedMeasuredDots, TransMeasurementToPerfect2, _ = getCorrectionTransform(CorrectedMeasuredDots, dotsGenInMeasuredFixed)

# sitk.WriteImage(dotsGenInMeasuredFixed, 'dotsGenInMeasured.nii')
# sitk.WriteImage(dotPatternMeasured, 'dotPatternMeasured.nii')
# sitk.WriteImage(CorrectedMeasuredDots, 'dotsFixed.nii')


# # Now chain these transforms and check again with squares
TransMeasurementToPerfect = list(squaresMeasuredToperfect) + list (TransMeasurementToPerfect)

for ix, trans in enumerate(TransMeasurementToPerfect):
    del(trans['InitialTransformParametersFileName'])
    sitk.WriteParameterFile(trans, fr'C:\OUTPUT\iQID Coreg\TransRectifyMeasurement_{ix}.txt')


##Generate a jacobian correction matrix
  
strx = sitk.TransformixImageFilter()
strx.ComputeDeterminantOfSpatialJacobianOn()
strx.SetTransformParameterMap(TransMeasurementToPerfect)
strx.SetMovingImage(dotPatternMeasured)
strx.SetOutputDirectory('.')
strx.Execute()

jacobian = sitk.ReadImage('spatialJacobian.nii')
sitk.WriteImage(jacobian, r'C:\OUTPUT\iQID Coreg\TransRectifyMeasurement_Jac.nii')

sitk.GetArrayFromImage(jacobian).mean()

corr = CorrectedDotsMeasured * jacobian

sitk.WriteImage(corr, 'JacobianCorrectionCameraWarp.nii')
# fixedSquares2 = sitk.Transformix(squaresGenInMeasured, TransMeasurementToPerfect)
# sitk.WriteImage(fixedSquares2, 'fixedSquares2.nii')



# squares = np.zeros_like(sitk.GetArrayViewFromImage(dotsGenInMeasured))
# for ix in range(squares.shape[0]):
#     for iy in range(squares.shape[1]):
#         if (ix%50 <2) or (iy%50 <2):
#             squares[ix, iy] = 1
# squares = sitk.GetImageFromArray(squares)
# squares.CopyInformation(dotsGenInMeasured)

# squaresWarped = sitk.Transformix(squares, TransMeasurementToPerfect)

# sitk.WriteImage(squares, 'squares.nii')
# sitk.WriteImage(squaresWarped, 'squaresWarped.nii')

# %%
