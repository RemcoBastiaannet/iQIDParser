from iQIDParser.iQIDParser import iQIDParser, pjoin, sitk
import matplotlib.pyplot as plt

fData = r'C:\Path\ToAlphaCamera\Listmode'

# load files
iQID = iQIDParser(fData)
alphaImg = iQID.generatePixelatedImage(imageScalingFactor=1)
npAlphaImg = sitk.GetArrayFromImage(alphaImg)

#Show the image
plt.figure()
plt.imshow(npAlphaImg, cmap='inferno', vmin=0, vmax = 0.5 * npAlphaImg.max())
plt.axis('off')
plt.tight_layout()

#Save the nifti image
sitk.WriteImage(alphaImg, 'alphaImg.nii')


