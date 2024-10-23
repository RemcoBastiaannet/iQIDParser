import os
from types import SimpleNamespace

import numpy as np
import SimpleITK as sitk

from iQIDParser.iQIDParser import iQIDParser, pjoin
from iQIDCoreg.iQIDCoreg import (
    doFullCoreg,
    doFullAnatomicalCoreg,
    wrapperGetCorrespondingSections,
)
from MicroscopeFileReader.MicroscopeFileReader import readTiff

from MicroscopeFileReader import ImageManipulation


# Run pars
dats1 = [
    {
        "fIQID": r"F:\DATA\iQID\5 May Batch 3D M1-3 tumors+1 kidney\Listmode",
        "fPhaloidin": r"F:\DATA\IXM\Run4\2023-08-08\5909\Stitched\Run4_A01_w2_TexasRed_stitched.tif",
        "fDAPI": r"F:\DATA\IXM\Run4\2023-08-08\5909\Stitched\Run4_A01_w1_DAPI_stitched.tif",
        "fTL": r"F:\DATA\IXM\Run4\2023-08-08\5909\Stitched\Run4_A01_w3_TL_stitched.tif",
        "fOutputDir": r"f:\tmpDump\Run4",
        "flipAxes": False,
    },
    {
        "fIQID": r"F:\DATA\iQID\5 May Batch 3D M1-3 tumors+1 kidney\Listmode",
        "fPhaloidin": r"F:\DATA\IXM\Run4a\2023-08-08\5910\Stitched\Run4a_A01_w2_TexasRed_stitched.tif",
        "fDAPI": r"F:\DATA\IXM\Run4a\2023-08-08\5910\Stitched\Run4a_A01_w1_DAPI_stitched.tif",
        "fTL": r"F:\DATA\IXM\Run4a\2023-08-08\5910\Stitched\Run4a_A01_w3_TL_stitched.tif",
        "fOutputDir": r"f:\tmpDump\Run4a",
        "flipAxes": False,
    },
    {
        "fIQID": r"F:\DATA\iQID\5 May Batch 3D M1-3 tumors+1 kidney\Listmode",
        "fPhaloidin": r"F:\DATA\IXM\Run4c\2023-08-08\5911\Stitched\Run4c_A01_w2_TexasRed_stitched.tif",
        "fDAPI": r"F:\DATA\IXM\Run4c\2023-08-08\5911\Stitched\Run4c_A01_w1_DAPI_stitched.tif",
        "fTL": r"F:\DATA\IXM\Run4c\2023-08-08\5911\Stitched\Run4c_A01_w3_TL_stitched.tif",
        "fOutputDir": r"f:\tmpDump\Run4c",
        "flipAxes": True,
    },
]


############################
## RUN 3 ###################
############################
dats2 = [
    {
        "fIQID": r"F:\DATA\iQID\2 May Batch M2 and M3 6h Tumor and Kidney",
        "fPhaloidin": r"F:\DATA\IXM\Run3\2023-08-08\5906\Stitched\Run3_A01_w2_TexasRed_stitched.tif",
        "fDAPI": r"F:\DATA\IXM\Run3\2023-08-08\5906\Stitched\Run3_A01_w1_DAPI_stitched.tif",
        "fTL": r"F:\DATA\IXM\Run3\2023-08-08\5906\Stitched\Run3_A01_w3_TexasRed_stitched.tif",
        "fOutputDir": r"f:\tmpDump\M3T1L3 KidL 2,5,9",
        "flipAxes": True,
    },
    {
        "fIQID": r"F:\DATA\iQID\2 May Batch M2 and M3 6h Tumor and Kidney",
        "fPhaloidin": r"F:\DATA\IXM\Run3\2023-08-08\5906\Stitched\Run3_A02_w2_TexasRed_stitched.tif",
        "fDAPI": r"F:\DATA\IXM\Run3\2023-08-08\5906\Stitched\Run3_A02_w1_DAPI_stitched.tif",
        "fTL": r"F:\DATA\IXM\Run3\2023-08-08\5906\Stitched\Run3_A02_w3_TexasRed_stitched.tif",
        "fOutputDir": r"f:\tmpDump\M2T1 KidL 2,11",
        "flipAxes": True,
    },
    {
        "fIQID": r"F:\DATA\iQID\2 May Batch M2 and M3 6h Tumor and Kidney",
        "fPhaloidin": r"F:\DATA\IXM\Run3\2023-08-08\5906\Stitched\Run3_A03_w2_TexasRed_stitched.tif",
        "fDAPI": r"F:\DATA\IXM\Run3\2023-08-08\5906\Stitched\Run3_A03_w1_DAPI_stitched.tif",
        "fTL": r"F:\DATA\IXM\Run3\2023-08-08\5906\Stitched\Run3_A03_w3_TexasRed_stitched.tif",
        "fOutputDir": r"f:\tmpDump\M3T1L1 TumR 2,4,6",
        "flipAxes": True,
    },
    {
        "fIQID": r"F:\DATA\iQID\2 May Batch M2 and M3 6h Tumor and Kidney",
        "fPhaloidin": r"F:\DATA\IXM\Run3\2023-08-08\5906\Stitched\Run3_A04_w2_TexasRed_stitched.tif",
        "fDAPI": r"F:\DATA\IXM\Run3\2023-08-08\5906\Stitched\Run3_A04_w1_DAPI_stitched.tif",
        "fTL": r"F:\DATA\IXM\Run3\2023-08-08\5906\Stitched\Run3_A04_w3_TexasRed_stitched.tif",
        "fOutputDir": r"f:\tmpDump\M2T1TumrR 2,4,6",
        "flipAxes": True,
    },
]


############################
## T1 ######################
############################
ixmBaseFolder = r"F:\DATA\IXM\M2T2L1-Tum\2023-08-07\5900\Stitched"
dats3 = [
    {
        "fIQID": r"F:\DATA\iQID\3 May Batch M1-3 1D Tumor\Listmode",
        "fPhaloidin": pjoin(ixmBaseFolder, r"M2T2L1-Tum_A01_w2_TexasRed_stitched.tif"),
        "fDAPI": pjoin(ixmBaseFolder, "M2T2L1-Tum_A01_w1_DAPI_stitched.tif"),
        "fTL": pjoin(ixmBaseFolder, "M2T2L1-Tum_A01_w3_TL_stitched.tif"),
        "fOutputDir": r"f:\tmpDump\M2T2L1 TumL 2,4,6",
        "flipAxes": False,
    },
]


ixmBaseFolder = r"F:\DATA\IXM\T1-Part1\2023-08-07\5901\Stitched"
dats4 = [
    {
        "fIQID": r"F:\DATA\iQID\1 May Batch M1 Left tumor 6h\Listmode",
        "fPhaloidin": pjoin(ixmBaseFolder, "L1-T1-Part1_A01_w2_TexasRed_stitched.tif"),
        "fDAPI": pjoin(ixmBaseFolder, "L1-T1-Part1_A01_w1_DAPI_stitched.tif"),
        "fTL": pjoin(ixmBaseFolder, "L1-T1-Part1_A01_w3_TL_stitched.tif"),
        "fOutputDir": r"f:\tmpDump\L1 2,4,6",
        "flipAxes": True,
    },
    {
        "fIQID": r"F:\DATA\iQID\1 May Batch M1 Left tumor 6h\Listmode",
        "fPhaloidin": pjoin(ixmBaseFolder, "L2-T1-Part1_A02_w2_TexasRed_stitched.tif"),
        "fDAPI": pjoin(ixmBaseFolder, "L2-T1-Part1_A02_w1_DAPI_stitched.tif"),
        "fTL": pjoin(ixmBaseFolder, "L2-T1-Part1_A02_w3_TL_stitched.tif"),
        "fOutputDir": r"f:\tmpDump\L2 2,4,6",
        "flipAxes": True,
    },
    {
        "fIQID": r"F:\DATA\iQID\1 May Batch M1 Left tumor 6h\Listmode",
        "fPhaloidin": pjoin(ixmBaseFolder, "L3-T1-Part1_A03_w2_TexasRed_stitched.tif"),
        "fDAPI": pjoin(ixmBaseFolder, "L3-T1-Part1_A03_w1_DAPI_stitched.tif"),
        "fTL": pjoin(ixmBaseFolder, "L3-T1-Part1_A03_w3_TL_stitched.tif"),
        "fOutputDir": r"f:\tmpDump\L3 2,4,7",
        "flipAxes": True,
    },
]


datsColl = [
    {
        "fIQID": r"F:\DATA\iQID\3 May Batch M1-3 1D Tumor\Listmode",
        "fPhaloidin": pjoin(
            r"F:\DATA\IXM\M2T2L1-Tum\2023-08-07\5900\Stitched",
            r"M2T2L1-Tum_A01_w2_TexasRed_stitched.tif",
        ),
        "fDAPI": pjoin(
            r"F:\DATA\IXM\M2T2L1-Tum\2023-08-07\5900\Stitched",
            r"M2T2L1-Tum_A01_w1_DAPI_stitched.tif",
        ),
        "fTL": pjoin(
            r"F:\DATA\IXM\M2T2L1-Tum\2023-08-07\5900\Stitched",
            r"M2T2L1-Tum_A01_w3_TL_stitched.tif",
        ),
        "fOutputDir": r"f:\tmpDump\M2T2L1 TumL 2,4,6",
        "flipAxes": False,
    },
    {
        "fIQID": r"F:\DATA\iQID\2 May Batch M2 and M3 6h Tumor and Kidney",
        "fPhaloidin": r"F:\DATA\IXM\Run3\2023-08-08\5906\Stitched\Run3_A02_w2_TexasRed_stitched.tif",
        "fDAPI": r"F:\DATA\IXM\Run3\2023-08-08\5906\Stitched\Run3_A02_w1_DAPI_stitched.tif",
        "fTL": r"F:\DATA\IXM\Run3\2023-08-08\5906\Stitched\Run3_A02_w3_TexasRed_stitched.tif",
        "fOutputDir": r"f:\tmpDump\M2T1 KidL 2,11",  # still no good
        "flipAxes": False,
    },
    {
        "fIQID": r"F:\DATA\iQID\5 May Batch 3D M1-3 tumors+1 kidney\Listmode",
        "fPhaloidin": r"F:\DATA\IXM\Run4a\2023-08-08\5910\Stitched\Run4a_A01_w2_TexasRed_stitched.tif",
        "fDAPI": r"F:\DATA\IXM\Run4a\2023-08-08\5910\Stitched\Run4a_A01_w1_DAPI_stitched.tif",
        "fTL": r"F:\DATA\IXM\Run4a\2023-08-08\5910\Stitched\Run4a_A01_w3_TL_stitched.tif",
        "fOutputDir": r"f:\tmpDump\Run4a",
        "flipAxes": False,
    },
]

# dats = []
# dats.extend(datsColl)
dats = [
    {
        "fIQID": r"F:\DATA\iQID\2 May Batch M2 and M3 6h Tumor and Kidney",
        "fPhaloidin": r"F:\DATA\IXM\Run3\2023-08-08\5906\Stitched\Run3_A02_w2_TexasRed_stitched.tif",
        "fDAPI": r"F:\DATA\IXM\Run3\2023-08-08\5906\Stitched\Run3_A02_w1_DAPI_stitched.tif",
        "fTL": r"F:\DATA\IXM\Run3\2023-08-08\5906\Stitched\Run3_A02_w3_TexasRed_stitched.tif",
        "fOutputDir": r"f:\tmpDump\M2T1 KidL 2,11",
        "flipAxes": True, #still no good!!!!
    },
]
# dats.extend(dats2)
# dats.extend(dats3)
# dats.extend(dats4)


for dat in dats:
    loc = SimpleNamespace(**dat)

    # Prep folder
    os.makedirs(loc.fOutputDir, exist_ok=True)

    print(loc.fIQID)

    # load files
    iQID = iQIDParser(loc.fIQID)
    alphaImg = iQID.generatePixelatedImage(imageScalingFactor=1)

    # load Microscope image
    phaloidin = readTiff(loc.fPhaloidin, loc.flipAxes)
    DAPI = readTiff(loc.fDAPI, loc.flipAxes)
    TL = readTiff(loc.fTL, loc.flipAxes)

    sitk.WriteImage(TL, pjoin(loc.fOutputDir, "TL.mhd"))
    sitk.WriteImage(phaloidin, pjoin(loc.fOutputDir, "phaloidin.mhd"))
    sitk.WriteImage(alphaImg, pjoin(loc.fOutputDir, "rawAlpha.mhd"))
    sitk.WriteImage(DAPI, pjoin(loc.fOutputDir, "DAPI.mhd"))

    _ = wrapperGetCorrespondingSections(
        alphaImg, phaloidin, loc.fOutputDir, ignoreExist=False
    )


for dat in dats:
    loc = SimpleNamespace(**dat)

    alphaImg = sitk.ReadImage(pjoin(loc.fOutputDir, "rawAlpha.mhd"))
    phaloidin = sitk.ReadImage(pjoin(loc.fOutputDir, "phaloidin.mhd"))
    # DAPI = sitk.ReadImage(pjoin(loc.fOutputDir, 'DAPI.mhd'))

    DAPI = readTiff(loc.fDAPI, loc.flipAxes)
    sitk.WriteImage(DAPI, pjoin(loc.fOutputDir, "DAPI.mhd"))

    TL = sitk.ReadImage(pjoin(loc.fOutputDir, "TL.mhd"))

    matchingCoords = wrapperGetCorrespondingSections(
        alphaImg, phaloidin, loc.fOutputDir
    )

    # coregister alpha and phaloidin
    alphaImg = doFullCoreg(alphaImg, phaloidin, matchingCoords, loc.fOutputDir)

    # alphaImg = sitk.ReadImage( pjoin(fOutputDir, 'fullAlphaImageInPhaloidin.mhd') )

    # Remove alpha fiduicial markers
    # alphaImg = ImageManipulation.removeFiducialMarkersAlpha(alphaImg)

    # Get bounding boxes for sections
    boundingBoxes = ImageManipulation.getObjectBoundingBoxes(phaloidin)

    phaloidinCuts = ImageManipulation.applyObjectBounsdingBoxes(
        phaloidin, boundingBoxes
    )

    DAPICuts = ImageManipulation.applyObjectBounsdingBoxes(DAPI, boundingBoxes)

    TLCuts = ImageManipulation.applyObjectBounsdingBoxes(TL, boundingBoxes)

    alphaImgCuts = ImageManipulation.applyObjectBounsdingBoxes(
        alphaImg,
        (
            np.array(boundingBoxes)
            * phaloidin.GetSpacing()[0]
            / alphaImg.GetSpacing()[0]
        ).astype(int),
    )

    ImageManipulation.saveImageList(phaloidinCuts, "PhaloidinCuts", loc.fOutputDir)
    ImageManipulation.saveImageList(DAPICuts, "DAPICuts", loc.fOutputDir)
    ImageManipulation.saveImageList(TLCuts, "TLCuts", loc.fOutputDir)
    ImageManipulation.saveImageList(alphaImgCuts, "alphaImgCuts", loc.fOutputDir)

    # # Coregister every section together
    # doFullAnatomicalCoreg(phaloidinCuts, DAPICuts,
    #                       alphaImgCuts, outputDir=fOutputDir)

    #  Create nice Composite images
    ImageManipulation.createComposites(
        phaloidinCuts, DAPICuts, alphaImgCuts, outputDir=loc.fOutputDir
    )
