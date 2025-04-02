# %%
from mplSettings import *
from tkinter import Tk  # from tkinter import Tk for Python 3.x
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
uci2bq = 37000

# Calibration Values
# Acts = [
#     0.068292683,
#     0.013658537,
#     0.006829268,
#     0.000136585,
#     6.82927e-05,
#     0.0001366,
#     0.0000683,
# ] #at 

# Acts = [i*uci2bq for i in Acts]


# Acts = [2.8, 0.56, 0.28, 0.056, 0.028, 0.0056, 0.0028]

Acts = [
    45.91388275,
    5.483495628,
    4.096206348,
    0.501557244,
    0.437694155,
    0.044029236,
    0.044740177,
]


# %%
# fData = r"C:\Calibration Curve Alpha Camera"
fData = r"C:\Calibration Curve New Scintillator"
iQID = iQIDParser(fData, listmodeType="Compressed")
iQID.updateReferenceTime(datetime(year=2024, month=11, day=22, hour=11, minute=20))

alphaImg_non_decay_corr = iQID.generatePixelatedImage(imageScalingFactor=1, decayCorrect=False)
alphaImg = iQID.generatePixelatedImage(imageScalingFactor=1, decayCorrect=True)

bq_ext = sitk.GetArrayFromImage(alphaImg_non_decay_corr).sum() / iQID.TotalSecondsMeasuring
bq_a = sitk.GetArrayFromImage(alphaImg).sum()

print(bq_a/sum(Acts))

sitk.WriteImage(alphaImg, pjoin(fData, "CalibrationImg.nii"))

# %% Assuming that there is a mask
from scipy.optimize import curve_fit

f = lambda x, a: a * x


def getR2(xdata, ydata, *popt):
    residuals = ydata - f(xdata, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared


alphaImg = sitk.ReadImage(pjoin(fData, "CalibrationImg.nii"))
# alphaImg = sitk.Median(alphaImg, (3, 3, 1))
sitk.WriteImage(sitk.Log10(alphaImg), pjoin(fData, "LogCalibrationImg.nii"))
# masks = sitk.ReadImage(pjoin(fData, "Smaller dots.seg.nrrd"))
masks = sitk.ReadImage(pjoin(fData, "Segmentation2.seg.nrrd"))
npMasks = np.squeeze(sitk.GetArrayFromImage(masks))
masks2 = sitk.GetImageFromArray(npMasks)
masks2.SetSpacing(masks.GetSpacing()[:2])
masks2.SetOrigin(masks.GetOrigin()[:2])

masks = sitk.Resample(masks2, alphaImg, sitk.Transform(), sitk.sitkNearestNeighbor)
npMasks = sitk.GetArrayFromImage(masks)
npAlpha = sitk.GetArrayFromImage(alphaImg)

#%%
counts = []
maskNumber = []
acts_accept = []
for ix, ival in enumerate(np.unique(npMasks)[1:]):
    # if ix == 0: continue
    # med = np.median(npAlpha[(npMasks == 1)])
    # counts.append(npAlpha[(npMasks == ival) * (npAlpha < 0.02)].sum())
    counts.append(npAlpha[(npMasks == ival)].sum())

    maskNumber.append(ival)
    acts_accept.append(Acts[ix])

counts = np.array(counts)
acts_accept = np.array(acts_accept)
# plt.figure()
# plt.imshow(npAlpha)
# plt.imshow(npMasks, alpha=0.5)
# plt.show()

# plt.figure()
# plt.plot(acts_accept,counts)
# counts = counts[1:]
# acts_accept = acts_accept[1:]

poptCalibration, _ = curve_fit(f, acts_accept, counts)
R2 = getR2(acts_accept, counts, *poptCalibration)


x = np.linspace(0, acts_accept.max(), 100)
#%%
plt.figure(figsize=(6,6))
plt.loglog(x, f(x, *poptCalibration), color = 'gray')
for i in range(len(acts_accept)):
    plt.scatter(acts_accept[i], counts[i], label=f"{acts_accept[i]}", color = 'k')

ax = plt.gca()
# Get the axis limits
xlim = ax.get_xlim()  # Get x-axis limits
ylim = ax.get_ylim()  # Get y-axis limits

# Find the bottom-right corner location
x = xlim[1]  # Rightmost x-coordinate
y = ylim[0]  # Bottommost y-coordinate

ax.text(x, y+.01, f'$R^2$ {R2:.3}', ha='right', va='bottom', fontsize=9)
ax.text(x, y, f'Detector efficiency 88.5%', ha='right', va='bottom', fontsize=9)

plt.xlabel('Calibrated activity [Bq]')
plt.ylabel('iQID measured activity [Bq]')
plt.savefig("__GenPlotsForPaper/CalibrationCurve.png", dpi=600)



for ix in range(len(acts_accept)):
    print(f"Act: {acts_accept[ix]}\tAlphaCamera: {counts[ix]}\t{(acts_accept[ix]-counts[ix])/acts_accept[ix] * 100}")
