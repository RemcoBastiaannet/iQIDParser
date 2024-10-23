from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename, askdirectory
import numpy as np
import SimpleITK as sitk
from os.path import join as pjoin

from iQIDParser.iQIDParser import iQIDParser
from iQIDCoreg.iQIDCoreg import doFullCoreg
from MicroscopeFileReader.MicroscopeFileReader import readTiff

from MicroscopeFileReader import ImageManipulation

Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing


fIQID = askdirectory(title='Select iQID listmode directory')
fPhaloidin = askopenfilename(title='Select Phaloidin scan [stitched]')
fDAPI = askopenfilename(title='Select DAPI scan [stitched]')
# show an "Open" dialog box and return the path to the selected file
fOutputDir = askdirectory(title='Select output directory', )


# %% load files
correctionTrans = [sitk.ReadParameterFile(fr'C:\OUTPUT\iQID Coreg\TransRectifyMeasurement_{ix}.txt') for ix in range(2)]

iQID = iQIDParser(fIQID)
alphaImg = iQID.generatePixelatedImage(imageScalingFactor=1)

sitk.WriteImage(alphaImg, pjoin(fOutputDir, 'alphaCameraImg.nii'))
alphaImg = sitk.Transformix(alphaImg, correctionTrans)
sitk.WriteImage(alphaImg, pjoin(fOutputDir, '_alphaCameraImg.nii'))

# load Microscope image
phaloidin = readTiff(fPhaloidin)
sitk.WriteImage(phaloidin, pjoin(fOutputDir, 'PhaloidinImage.nii'))

DAPI = readTiff(fDAPI)

# coregister alpha and phaloidin
alphaImg = doFullCoreg(alphaImg, phaloidin) #resulting in alphaImg in phaloidin
sitk.WriteImage(alphaImg, pjoin(fOutputDir, 'alphaCameraImgInMicroscopy.nii'))

# Remove alpha fiduicial markers
# alphaImg = ImageManipulation.removeFiducialMarkersAlpha(alphaImg)

# Get bounding boxes for sections
boundingBoxes = ImageManipulation.getObjectBoundingBoxes(phaloidin)

phaloidinCuts = ImageManipulation.applyObjectBounsdingBoxes(
    phaloidin, boundingBoxes)
DAPICuts = ImageManipulation.applyObjectBounsdingBoxes(DAPI, boundingBoxes)

alphaImgCuts = ImageManipulation.applyObjectBounsdingBoxes(
    alphaImg, (np.array(boundingBoxes) * phaloidin.GetSpacing()[0] / alphaImg.GetSpacing()[0]).astype(int))

# # Coregister every section together
# doFullAnatomicalCoreg(phaloidinCuts, DAPICuts,
#                       alphaImgCuts, outputDir=fOutputDir)

#  Create nice Composite images
ImageManipulation.createComposites(
    phaloidinCuts, DAPICuts, alphaImgCuts, outputDir=fOutputDir)
