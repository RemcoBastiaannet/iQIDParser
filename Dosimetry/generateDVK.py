
#%%
import numpy as np
import matplotlib.pyplot as plt


fDVK = r'F:\DATA\DVKBins.txt'

x = np.arange(100)

with open(fDVK, 'r') as f:
    dat = f.readlines()

dat = np.array([float(i.strip('\n')) for i in dat])
datNorm = dat / 1E5


# plt.figure()
# plt.semilogy(x, datNorm)
# plt.xlabel('Distance [$\mu$m]')
# plt.ylabel('Mean Energy Deposited [MeV]')
# plt.savefig('DPKProfile.png', dpi=600)

#%% TODO: DPK to DVK
import SimpleITK as sitk
from scipy.interpolate import interp1d


Kernel2D = np.zeros((101,101,51))
Kernel2D[50,50,25] = 1

kern2d = sitk.GetImageFromArray(Kernel2D.astype(int))
# kern2d.SetSpacing((14., 53.05/2,53.05/2.))
kern2d.SetSpacing((12., 12., 12.))

distSTK = sitk.SignedDanielssonDistanceMap(kern2d, False, False, True)
dist = sitk.GetArrayFromImage(distSTK)

interp = interp1d(x, datNorm, fill_value=0,bounds_error=False)

vals = interp(dist.flatten()).reshape(dist.shape)

distSteps = np.unique(np.floor(dist.flatten()))
for ix, i in enumerate(distSteps):
    
    vals[dist == i] /= np.sum((dist>= i)*(dist<distSteps[i+1]))


vals /= vals.flatten().sum() / datNorm.sum() 



plt.figure()
plt.imshow(vals[:,:,25]/vals.sum(), cmap = 'inferno')
plt.axis('off')
plt.tight_layout()


DVK = sitk.GetImageFromArray(vals)
sitk.WriteImage(DVK, 'DVK_Ac225_26_26_14.nii')
# %%
