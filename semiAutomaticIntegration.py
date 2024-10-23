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

fOutDir = r"C:\OUTPUT\iQID Coreg"
fAlphaCameraDir = r"F:\DATA\iQID\September 2024"


scalingFactor = 1.0
micScalingFactor = 1 / 10.

# %% Asking for data location

fIQID = askdirectory(title="Select iQID listmode directory", initialdir=fAlphaCameraDir)
# fIQID = r'F:\DATA\iQID\September 2024\September 2024 Timepoint 1 batch 2'
fiQIDData = pjoin(fIQID, "Listmode")

fMicroscopyFile = askopenfilename(
    title="Select Microscopy file", initialdir=fMicroscopyDir
)
# fMicroscopyFile = r'c:\Users\remco\Box\Lab Data\Sep 2024\Alpha Camera Slides\6H-048-Kidney-7-9-11.czi'
fSampleName = os.path.basename(os.path.normpath(fMicroscopyFile)).split(".czi")[0]

fOutputDir = askdirectory(title="Select OUTPUT directory", initialdir=fOutDir)
# fOutputDir = r'c:\OUTPUT\iQID Coreg\September 2024\6H-048-Kidney-7-9-11'
fOutputDir = pjoin(fOutputDir, fSampleName)
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
]  # TODO: fix this for our camera



os.makedirs(fOutputDir, exist_ok=True)

iQID = iQIDParser(fiQIDData, listmodeType="Compressed")

alphaImgLowRes = iQID.generatePixelatedImage(
    imageScalingFactor=scalingFactor, decayCorrect=False
)
alphaImgHiRes = iQID.generatePixelatedImage(imageScalingFactor=2, decayCorrect=True)

sitk.WriteImage(alphaImgHiRes, pjoin(fOutputDir, "alphaImgHiRes.nii"))

for ix in range(len(correctionTrans)):
    locSpacing = [float(i) for i in correctionTrans[ix]["Spacing"]]
    correctionTrans[ix]["Spacing"] = [str(i) for i in alphaImgHiRes.GetSpacing()]

    sizeFac = [
        float(locSpacing[ix] / float(correctionTrans[ix]["Spacing"][ix]))
        for ix in range(len(locSpacing))
    ]
    correctionTrans[ix]["Size"] = [str(i) for i in alphaImgHiRes.GetSize()]

# alphaImgHiRes = sitk.Transformix(alphaImgHiRes, correctionTrans)
sitk.WriteImage(alphaImgHiRes, pjoin(fOutputDir, "alphaImgHiRes2_cor.nii"))


sitk.WriteImage(alphaImgLowRes, pjoin(fOutputDir, "alphaImgLowRes.nii"))

for ix in range(len(correctionTrans)):
    locSpacing = [float(i) for i in correctionTrans[ix]["Spacing"]]
    correctionTrans[ix]["Spacing"] = [str(i) for i in alphaImgLowRes.GetSpacing()]

    sizeFac = [
        float(locSpacing[ix] / float(correctionTrans[ix]["Spacing"][ix]))
        for ix in range(len(locSpacing))
    ]
    correctionTrans[ix]["Size"] = [str(i) for i in alphaImgLowRes.GetSize()]


# alphaImgLowRes = sitk.Transformix(alphaImgLowRes, correctionTrans)
sitk.WriteImage(alphaImgLowRes, pjoin(fOutputDir, "alphaImgLowRes_corr.nii"))

alphaImgLowRes = sitk.ReadImage(pjoin(fOutputDir, "alphaImgLowRes_corr.nii"))

matchingcoords = wrapperGetCorrespondingSections(
    alphaImgLowRes, Phalloidin, fOutputDir, ignoreExist=True
)


with open(pjoin(fOutputDir, "matchingCoords.pic"), "rb") as f:
    matchingcoords = pickle.load(f)


DAPIinAlpha, phaloidininAlpha = doFullCoreg(
    alphaImgLowRes, DAPI, Phalloidin, matchingCoords=matchingcoords, outputDir=fOutputDir
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

# Get bounding boxes for sections
boundingBoxes = ImageManipulation.getObjectBoundingBoxes(phaloidininAlpha + DAPIinAlpha)

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
    phaloidinCuts, DAPICuts, alphaImgCuts, outputDir=fOutputDir
)

print(f"Succesfully completed {fSampleName}")
