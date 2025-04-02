from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename, askdirectory
from tkinter.simpledialog import askstring
import numpy as np
import SimpleITK as sitk
from os.path import join as pjoin
import os

from skimage.restoration import rolling_ball
from skimage.filters import threshold_otsu
import fill_voids

from iQIDParser.iQIDParser import iQIDParser
from iQIDCoreg.iQIDCoreg import doFullCoreg, wrapperGetCorrespondingSections
from MicroscopeFileReader.MicroscopeFileReader import readTiff, getPixelSize

from MicroscopeFileReader import ImageManipulation

import pickle

import matplotlib.pyplot as plt
from datetime import datetime

Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing


#%% Asking for data location
fIQIDDataFolder = askdirectory(title='Select directory containing all iQID recordings')

for iFile in os.listdir(fIQIDDataFolder):
    if os.path.isfile(iFile): continue
    if 'linearity' in iFile.lower(): continue

    candidateListModeFolder = pjoin(fIQIDDataFolder, iFile, 'Listmode')
    
    if not os.path.isdir(candidateListModeFolder):  continue

    print( f"Processing {iFile}" )
    try:
        iQID = iQIDParser(candidateListModeFolder, listmodeType="Compressed")
    

        alphaImg = iQID.generatePixelatedImage(imageScalingFactor=1, decayCorrect=False)
        alphaImg = sitk.Cast(alphaImg, sitk.sitkFloat32)
        #Write Image to containing data folder
        sitk.WriteImage(alphaImg, pjoin(fIQIDDataFolder, iFile, 'AlphaImg.tiff'))
        sitk.WriteImage(sitk.Log(alphaImg+1), pjoin(fIQIDDataFolder, iFile, 'AlphaImg_log.tiff'))



    except:
        print( f"Failed at {iFile}" )