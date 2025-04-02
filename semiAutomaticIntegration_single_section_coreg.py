from tkinter import Tk  # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askdirectory, askopenfilename

import numpy as np
import SimpleITK as sitk
from os.path import join as pjoin
import os

from iQIDParser import iQIDParser
from iQIDCoreg.iQIDCoreg import doFullCoreg, wrapperGetCorrespondingSections
from MicroscopeFileReader.MicroscopeFileReader import readTiff, getPixelSize

from MicroscopeFileReader import ImageManipulation

import pickle
from aicspylibczi import CziFile


Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing

fMicroscopyDir = r"C:\Users\remco\Box\Lab Data\Sep 2024\Alpha Camera Slides"

fOutputDir = r'c:\OUTPUT\iQID Coreg\September 2024'
fAlphaCameraDir = r"F:\DATA\iQID\September 2024"
fSampleName = r'T2_RTum_064_13_15_17'
# fMicroscopyFile = r'C:\OUTPUT\iQID Coreg\September 2024\T3_LTum_031singleSectionMatch_section3\T3_LTum_031_7_9_11-Scene-3-ScanRegion2.czi'
fMicroscopyFile = r'C:\OUTPUT\iQID Coreg\September 2024\T2_RTum_064_13_15_17_SingleMatchingSection\01_T2_064_RTum_13_15_17_AlphaCamera_Phalloidin_Hoechst_10X-Scene-1-ScanRegion0.czi'
scalingFactor = 1
micScalingFactor = 1 / 10.

# %% Asking for data location


# fOutputDir = r'c:\OUTPUT\iQID Coreg\September 2024\6H-048-Kidney-7-9-11'
fOutputDir = pjoin(fOutputDir, fSampleName+'singleSectionMatch_section3')# + '_without_warping_corr')
os.makedirs(fOutputDir, exist_ok=True)

# %% Loading files
micFile = CziFile(fMicroscopyFile)

# Extract physical pixel sizes
scaling = micFile.meta.findall('.//Scaling/Items/Distance')
MicroscopePixelSpacing = [float(scaling[0].find('Value').text) * 1E6, float(scaling[1].find('Value').text) *1E6]



DAPI = sitk.GetImageFromArray(micFile.read_mosaic(C=0, scale_factor=micScalingFactor)[0,::])
DAPI.SetSpacing([i / micScalingFactor for i in  MicroscopePixelSpacing])

Phalloidin = sitk.GetImageFromArray(micFile.read_mosaic(C=1, scale_factor=micScalingFactor)[0,::])
Phalloidin.SetSpacing([i / micScalingFactor for i in  MicroscopePixelSpacing])

#Correct for intensity changes
# Phalloidin = sitk.AdaptiveHistogramEqualization(Phalloidin)
# DAPI = sitk.AdaptiveHistogramEqualization(DAPI)

correctionTrans = [
    sitk.ReadParameterFile(rf"C:\OUTPUT\iQID Coreg\TransRectifyMeasurement_{ixs}.txt")
    for ixs in range(5)
]  

correction_Jac = sitk.ReadImage(r'C:\OUTPUT\iQID Coreg\TransRectifyMeasurement_Jac.nii')


os.makedirs(fOutputDir, exist_ok=True)


###########################
## Generate Alpha Hi res ##
###########################

alphaImgHiRes = sitk.ReadImage(r'C:\OUTPUT\iQID Coreg\September 2024\T2_RTum_064_13_15_17\alphaImgHiRes2_cor.nii')
sitk.WriteImage(alphaImgHiRes, pjoin(fOutputDir, "alphaImgHiRes.nii"))
############################
## Generate alpha low res ##
############################

alphaImgLowRes = sitk.ReadImage(r'C:\OUTPUT\iQID Coreg\September 2024\T2_RTum_064_13_15_17\alphaImgLowRes_corr.nii')
sitk.WriteImage(alphaImgLowRes, pjoin(fOutputDir, "alphaImgLowRes.nii"))
#######################################
## Get correspondence between images ##
#######################################

matchingcoords = wrapperGetCorrespondingSections(
    alphaImgHiRes, Phalloidin, fOutputDir, ignoreExist=False
)


with open(pjoin(fOutputDir, "matchingCoords.pic"), "rb") as f:
    matchingcoords = pickle.load(f)


DAPIinAlpha, phaloidininAlpha = doFullCoreg(
    alphaImgHiRes, DAPI, Phalloidin, matchingCoords=matchingcoords, outputDir=fOutputDir
)  


refImg = sitk.Image(alphaImgHiRes)
refImg.SetSpacing(DAPIinAlpha.GetSpacing())

resampler = sitk.ResampleImageFilter()
resampler.SetOutputOrigin(alphaImgHiRes.GetOrigin())
resampler.SetOutputDirection(alphaImgHiRes.GetDirection())
resampler.SetOutputSpacing(DAPIinAlpha.GetSpacing())
newSize = [
    alphaImgHiRes.GetSize()[i]
    * alphaImgHiRes.GetSpacing()[i]
    / DAPIinAlpha.GetSpacing()[i]
    for i in range(DAPIinAlpha.GetDimension())
]
newSize = [int(i) for i in newSize]
resampler.SetSize(newSize)
DAPIinAlpha = resampler.Execute(DAPIinAlpha)
phaloidininAlpha = resampler.Execute(phaloidininAlpha)


##########
# Get bounding boxes for sections
##########

boundingBoxes = ImageManipulation.getObjectBoundingBoxes(phaloidininAlpha + DAPIinAlpha, numSections=1)

phaloidinCuts = ImageManipulation.applyObjectBounsdingBoxes(
    phaloidininAlpha, boundingBoxes
)


DAPICuts = ImageManipulation.applyObjectBounsdingBoxes(DAPIinAlpha, boundingBoxes)

alphaImgCutsHiRes = ImageManipulation.applyObjectBounsdingBoxes(
    alphaImgHiRes, np.array(boundingBoxes)
)

for ixt, iAlpha in enumerate(alphaImgCutsHiRes):
    sitk.WriteImage(iAlpha, pjoin(fOutputDir, f"alphaImgHiRes_{ixt}.nii"))


alphaImgCuts = ImageManipulation.applyObjectBounsdingBoxes(
    alphaImgLowRes, np.array(boundingBoxes)
)

for ixt, iAlpha in enumerate(alphaImgCuts):
    sitk.WriteImage(iAlpha, pjoin(fOutputDir, f"alphaImgLowRes_{ixt}.nii"))


# # Coregister every section together
# doFullAnatomicalCoreg(phaloidinCuts, DAPICuts,
#                       alphaImgCuts, outputDir=fOutputDir)

#  Create nice Composite images
ImageManipulation.createComposites(
    phaloidinCuts, DAPICuts, alphaImgCutsHiRes, outputDir=fOutputDir
)
print(f"Succesfully completed {fSampleName}")
