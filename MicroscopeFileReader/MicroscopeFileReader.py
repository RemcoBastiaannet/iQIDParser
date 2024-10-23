from aicsimageio.aics_image import AICSImage
import SimpleITK as sitk
import numpy as np

def resample_volume(
    volume, interpolator=sitk.sitkLinear, new_spacing=[0.39, 0.39, 0.55]
):
    volume = sitk.Cast(volume, sitk.sitkFloat32)  # read and cast to float32

    original_spacing = volume.GetSpacing()

    # new_spacing = list(new_spacing)
    # new_spacing[-1] = original_spacing[-1]

    # volumeDifferenceCorrectionFactor = 1#np.prod(new_spacing[:-1])/np.prod(original_spacing[:-1])
    # print(f"VolumeCorrectionFactor: {volumeDifferenceCorrectionFactor}")
    original_size = volume.GetSize()
    new_size = [
        int(round(osz * ospc / nspc))
        for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)
    ]
    return sitk.Resample(
        volume,
        new_size,
        sitk.Transform(),
        interpolator,
        volume.GetOrigin(),
        new_spacing,
        volume.GetDirection(),
        0,
        volume.GetPixelID(),
    )


def readTiff(fFile: str, standardPixelSize: None|list = None, flipAxes: bool = False, rescalingFac = 1) -> sitk.Image:

    micData = AICSImage(fFile)

    pixelsizes = micData.physical_pixel_sizes[1:][::-1]
    npMicImg = np.squeeze(micData.get_image_data())
    if flipAxes:
        npMicImg = npMicImg[:, ::-1]
        
    MicImg = sitk.GetImageFromArray(npMicImg)

    if standardPixelSize is not None:
        pixelsizes = standardPixelSize

    # pixelsizes = [i*1E2 for i in pixelsizes]

    MicImg.SetSpacing(pixelsizes)

    if rescalingFac != 1:
        newspacing = [i / rescalingFac for i in pixelsizes]
        MicImg = resample_volume( MicImg, interpolator=sitk.sitkLinear, new_spacing=newspacing)

    return MicImg

def getPixelSize(name: str):

    standardPixelSize5x = [1.614]*2 + [1.]
    # standardPixelSize5x = [1.5]*2 + [1.]
    standardPixelSize10x = [0.6356]*2 + [1.]

    standardPixelSize = standardPixelSize5x

    if name == 'C3M2T1Kid':
        standardPixelSize = standardPixelSize10x
    elif name == 'C3M1T1Kid':
        standardPixelSize = standardPixelSize10x
    elif name == 'C3M1T1Tum':
        standardPixelSize = standardPixelSize10x
    elif name == 'C3M2T1Kid':
        standardPixelSize = standardPixelSize10x
    elif name == 'C3M2T1Tum':
        standardPixelSize = standardPixelSize5x
    elif name == 'C3M3T1Kid':
        standardPixelSize = standardPixelSize10x
    elif name == 'C3M3T1Tum':
        standardPixelSize = standardPixelSize5x
    elif name == 'C4M1T3Kid':
        standardPixelSize = standardPixelSize10x
    elif name == 'C4M1T3Tum':
        standardPixelSize = standardPixelSize5x
    elif name == 'C4M2T3Kid':
        standardPixelSize = standardPixelSize5x
    elif name == 'C4M2T3Tum':
        standardPixelSize = standardPixelSize5x
    elif name == 'C4M3T3Kid':
        standardPixelSize = standardPixelSize5x
    elif name == 'C4M3T3Tum':
        standardPixelSize = standardPixelSize10x
    elif name == 'C4M4T4Kid':
        standardPixelSize = standardPixelSize5x     
    elif name == 'C4M4T4Tum':
        standardPixelSize = standardPixelSize5x 
    elif name == 'C5M1T2Tum':
        standardPixelSize = standardPixelSize10x
    elif name == 'C5M2T2Kid':
        standardPixelSize = standardPixelSize10x
    elif name == 'C5M2T2Tum':
        standardPixelSize = standardPixelSize5x
    elif name == 'C5M3T2Kid':
        standardPixelSize = standardPixelSize5x
    elif name == 'C5M3T2Tum':
        standardPixelSize = standardPixelSize5x
    elif name == 'C5M4T4Kid':
        standardPixelSize = standardPixelSize10x
    elif name == 'C5M5T4Kid':
        standardPixelSize = standardPixelSize10x
    
    else: 
        standardPixelSize = standardPixelSize5x
    
    return standardPixelSize