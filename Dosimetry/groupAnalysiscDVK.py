import os
from os.path import join as pjoin

import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

from Dosimetry import get_cDVH


fDataDir = r'C:\OUTPUT\iQID Coreg\September 2024'

cDVHs, cDVH_acts, xDosess, timePoints, structures = [], [], [], [], []
for i in os.listdir(fDataDir):
    if 'Hot Kidney Phalloidin&Hoechst' in i: continue
    if 'Kid' in i: continue

    f = pjoin(fDataDir, i)

    if not os.path.isdir(f): continue
    
    # check if all the files exist
    fHE = pjoin(f, 'CentralHEIntensityMap.nii')
    fDoseRate = pjoin(f, 'DoseRateMap.nii')
    fDoseRate_act = pjoin(f, 'DoseRateMap_basedon_activityRecon.nii')

    if not os.path.isfile(  fHE ): continue
    if not os.path.isfile( fDoseRate ): continue
    if not os.path.isfile( fDoseRate_act ): continue

    print(f)

    HE = sitk.ReadImage(fHE) 
    DR = sitk.ReadImage(fDoseRate) 
    DR_act = sitk.ReadImage(fDoseRate_act) 

    DR_sm = sitk.GetArrayViewFromImage(DR_act).sum()
    DR_act_sm = sitk.GetArrayViewFromImage(DR_act).sum()

    DR = sitk.Resample(DR, HE, sitk.Transform(), sitk.sitkLinear)
    DR_act = sitk.Resample(DR_act, HE, sitk.Transform(), sitk.sitkLinear)

    npHE = sitk.GetArrayFromImage(HE)
    npDR = sitk.GetArrayFromImage(DR)
    npDR_act = sitk.GetArrayFromImage(DR_act)


    npDR /= npDR.sum()
    npDR *= DR_sm

    npDR *= 1E3 #Gy/s to milliGy/s
    npDR *= (60*60) #cGy/h
    
    npDR_act /= npDR_act.sum()
    npDR_act *= DR_act_sm

    npDR_act *= 1E3 #Gy/s to milliGy/s
    npDR_act *= (60*60) #cGy/h
    
    xDoses, cDVH = get_cDVH(npDR, npHE, max_dose = 10., normalize = False)
    xDoses, cDVH_act = get_cDVH(npDR_act, npHE, max_dose = 10., normalize = False)

    timePoint = i.split('_')[0][1:]
    structure = i.split('_')[1]

    cDVHs.append(cDVH)
    cDVH_acts.append(cDVH_act)
    timePoints.append(timePoint)
    structures.append(structure)


#%% Plot combined results


diffs = []

for ix in range(len(cDVHs)):
    diffs.append( cDVHs[ix] - cDVH_acts[ix] )
    # plt.plot(xDoses, cDVHs[ix] - cDVH_acts[ix], label = f'{timePoints[ix]}')
    # plt.plot(xDoses, cDVH_acts[ix]*100, color = 'r')
    # plt.plot(xDoses, cDVHs[ix]*100, color = 'k')

med_DVH = np.median(np.array(cDVHs),0)
med_DVH_act = np.median(np.array(cDVH_acts),0)


plt.figure()
plt.plot(xDoses, med_DVH*100, color = 'k', linewidth = 3)
plt.xlabel('Dose rate [mGy/h]')
plt.ylabel('Median relative volume [%]')
# plt.savefig('MediancDVH_tumors.png', dpi = 600)

plt.figure()
plt.axhline(0, color = 'lightgray')
plt.plot(xDoses, np.median(np.array(diffs),0) * 100, linewidth = '3')

plt.legend()
plt.xlabel('Dose rate [mGy/h]')
plt.ylabel('Median difference [$cDVH_{anatomical ref} - cDVH_{activity}$ %]')
# plt.savefig('MediancDVH_Anatomy_is_generally_higher than activity_based only.png', dpi = 600)

#%% Plot cDVH per timepoint
plt.figure()
cDVHs = np.array(cDVHs)
for iT in np.unique(timePoints):
    ix = [ix for ix, i in enumerate(timePoints) if i == iT]

    locMn = np.mean(cDVHs[ix,:], axis = 0)

    plt.plot(xDoses, locMn, label = f'Time point {iT}')

plt.legend()
plt.show()


