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
from skimage import filters
from mplSettings import *
from scipy.ndimage import distance_transform_edt

fMainDir = r'C:\OUTPUT\iQID Coreg\September 2024'


def getTissueMask(HEKidney: sitk.Image) -> sitk.Image:
    # label_map = sitk.OtsuMultipleThresholds(HEKidney, 2, 0, 256, True)
    # tissueMask = label_map !=4
    tissueMask = HEKidney < .95
    tissueMask = sitk.BinaryFillhole(tissueMask, fullyConnected=True)
    return tissueMask


def get_cDVH(npAlphaImg: np.ndarray, npHE: np.ndarray, max_dose: float = 4., normalize = True):
    #Get Tissue mask
    
    # blurred_image = filters.gaussian(npHE, sigma = 10)
    # tissue_thresh = filters.threshold_otsu(blurred_image)
    # msk = blurred_image < tissue_thresh
    msk = sitk.GetArrayFromImage(getTissueMask(sitk.GetImageFromArray(npHE)))

    #Construct cDVH
    doseRateVals = npAlphaImg[msk!=0].flatten()
    xDoses = np.linspace(0, max_dose, 100)
    cDVH = np.zeros_like(xDoses)

    norm_fac = 1
    if normalize: norm_fac = len(doseRateVals)

    for ix in range(len(xDoses)):
        cDVH[ix] = np.sum( doseRateVals >= xDoses[ix] ) / norm_fac

    return xDoses, cDVH

#Crawl over all dose rate maps that we have -> plot as function of timepoint & normalize to see changing shape

#%%
# plt.figure()
# styles = ['-', '--', '-.', ':']
# plt.gca().set_prop_cycle(cycler('linestyle', styles))
for iDir in os.listdir(fMainDir):

    if 'tum' not in iDir.lower(): continue
    Timepoint = [i for i in iDir.split('_') if i[0] == 'T' and i[1].isnumeric()]
    if len(Timepoint) == 0: continue
    
    num = Timepoint[0][1:]

    fDoserate = pjoin(fMainDir, iDir, 'DoseRateMap.nii')
    fHE = pjoin(fMainDir, iDir, 'CentralHEIntensityMap.nii')

    if not os.path.isfile(fDoserate): continue
    if not os.path.isfile(fHE): continue

    HE = sitk.ReadImage(fHE)
    npHE = sitk.GetArrayFromImage(HE)
    
    # msk = getTissueMask(HE)
    msk = sitk.OtsuThreshold(HE)
    msk = sitk.BinaryFillhole(msk)

    doseRate = sitk.ReadImage(fDoserate)
    doseRate = sitk.Resample(doseRate, HE, sitk.Transform(), sitk.sitkLinear)
    npDoseRate = sitk.GetArrayFromImage(doseRate)
    npDoseRate /= npDoseRate.sum()
    npDoseRate *= sitk.GetArrayViewFromImage(doseRate).sum()

    npDoseRate *= 1E3 #Gy/s to milliGy/s
    npDoseRate *= (60*60) #cGy/h

    # plt.figure()
    # plt.imshow(npDoseRate)
    # plt.show()


    activityF = npDoseRate.flatten()

    dists = distance_transform_edt(sitk.GetArrayFromImage(msk))

    dists[sitk.GetArrayViewFromImage(msk) == 0] += 10000000

    maxdist  = 200
    distsF = dists.flatten()
    selIX = distsF < maxdist
    
    uniqueDists = np.unique(np.round(distsF[selIX]))
    avgAct = np.zeros_like(uniqueDists)
    for ix, iUn in enumerate(uniqueDists):
        selLocs = (distsF == iUn)
        avgAct[ix] = np.nanmean(activityF[selLocs])

    plt.figure()
    plt.plot(uniqueDists, avgAct)
    plt.xlim([0, 200])
    plt.title(num)


    # maxDose = npDoseRate.max()
    # xDoses, cDVH_activity = get_cDVH(npDoseRate, npHE, maxDose)

    # print(num)
    # plt.plot(xDoses, cDVH_activity*100., color = f'C{int(num)-2}', linewidth = 2)

# plt.xlim([0, 100])
# plt.show()

    # sampleDir = pjoin(fMainDir, iDir)