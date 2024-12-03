from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename, askdirectory
import numpy as np
import SimpleITK as sitk
from os.path import join as pjoin

from iQIDParser.iQIDParser import iQIDParser
from iQIDCoreg.iQIDCoreg import doFullCoreg
from MicroscopeFileReader.MicroscopeFileReader import readTiff

from MicroscopeFileReader import ImageManipulation

Tk().withdraw()

############## NZ: Required changes for PAS stain ############################################################################################

fIQID = askdirectory(title='Select iQID listmode directory')
fPhaloidin = askopenfilename(title='Select Phaloidin scan [stitched]')
fDAPI = askopenfilename(title='Select DAPI scan [stitched]')
fOutputDir = askdirectory(title='Select output directory', )
#################
# fIQID = askdirectory(title='Select iQID listmode directory')
# fPAS = askopenfilename(title='Select PAS scan [stitched]')
# fOutputDir = askdirectory(title='Select output directory', )
##########################################################################################################################################


correctionTrans = [sitk.ReadParameterFile(fr'C:\OUTPUT\iQID Coreg\TransRectifyMeasurement_{ix}.txt') for ix in range(2)]

iQID = iQIDParser(fIQID)
alphaImg = iQID.generatePixelatedImage(imageScalingFactor=1)

sitk.WriteImage(alphaImg, pjoin(fOutputDir, 'alphaCameraImg.nii'))
alphaImg = sitk.Transformix(alphaImg, correctionTrans)
sitk.WriteImage(alphaImg, pjoin(fOutputDir, '_alphaCameraImg.nii'))



############## NZ: Required changes for PAS stain ############################################################################################
phaloidin = readTiff(fPhaloidin)
sitk.WriteImage(phaloidin, pjoin(fOutputDir, 'PhaloidinImage.nii'))
DAPI = readTiff(fDAPI)
#####################
# PAS = readTiff(fPAS)
# sitk.WriteImage(PAS, pjoin(fOutputDir, 'PASImage.nii'))
#############################################################################################################################################



############## NZ: Required changes for PAS stain ############################################################################################
alphaImg = doFullCoreg(alphaImg, phaloidin)
sitk.WriteImage(alphaImg, pjoin(fOutputDir, 'alphaCameraImgInMicroscopy.nii'))

boundingBoxes = ImageManipulation.getObjectBoundingBoxes(phaloidin)

phaloidinCuts = ImageManipulation.applyObjectBounsdingBoxes(phaloidin, boundingBoxes)
DAPICuts = ImageManipulation.applyObjectBounsdingBoxes(DAPI, boundingBoxes)

alphaImgCuts = ImageManipulation.applyObjectBounsdingBoxes(alphaImg, (np.array(boundingBoxes) * phaloidin.GetSpacing()[0] / alphaImg.GetSpacing()[0]).astype(int))

ImageManipulation.createComposites(phaloidinCuts, DAPICuts, alphaImgCuts, outputDir=fOutputDir)
##################
# alphaImg = doFullCoreg(alphaImg, PAS) 
# sitk.WriteImage(alphaImg, pjoin(fOutputDir, 'alphaCameraImgInMicroscopy.nii'))
# boundingBoxes = ImageManipulation.getObjectBoundingBoxes(phaloidin)

#PASCuts = ImageManipulation.applyObjectBounsdingBoxes(PAS, boundingBoxes)
#alphaImgCuts = ImageManipulation.applyObjectBounsdingBoxes(alphaImg, (np.array(boundingBoxes) * PAS.GetSpacing()[0] / alphaImg.GetSpacing()[0]).astype(int))

#ImageManipulation.createComposites(PASCuts, alphaImgCuts, outputDir=fOutputDir)

#############################################################################################################################################
