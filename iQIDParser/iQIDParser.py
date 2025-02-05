import os
from os.path import join as pjoin
from datetime import datetime
import copy

import numpy as np
import SimpleITK as sitk


from scipy.signal import medfilt

from .iQIDListModeFile import iQIDListModeFile
from .constants import *

class iQIDParser:

    fileList: list[str] = []
    parsedListModeFiles: list[iQIDListModeFile] = []
    imageSizeFactor = 1
    lambdaDays = lambdaAc
    imageSpacing = [alphaCameraPixSize] * 2
    referenceTime = datetime(2023, 1, 1)
    cameraSensitivity = cameraSensitivity
    decayCorrectionFactor = 1

    def __init__(self, path: str | list[str], listmodeType: str = "Compressed"):

        self.parsedListModeFiles = []

        if isinstance(path, str):
            for root, _, files in os.walk(path):
                for f in files:
                    if listmodeType in f:
                        self.fileList.append(pjoin(root, f))
                        self.parsedListModeFiles.append(
                            iQIDListModeFile(pjoin(root, f))
                        )

            self.parsedListModeFiles = [
                i for i in self.parsedListModeFiles if i.rowCentroids is not None
            ]

        elif isinstance(path, list):
            for f in path:
                self.fileList.append(f)
                self.parsedListModeFiles.append(iQIDListModeFile(f))

        self.updateReferenceTime()

    def updateReferenceTime(self, refTime: datetime | None = None):

        if refTime is None:  # Set to the beginning of the first file
            startTimes = [i.fileStartDate for i in self.parsedListModeFiles]
            self.referenceTime = min(startTimes)

        else:
            self.referenceTime = refTime

        self.updateDecayCorrectionScaling()

    def getScaledDecayNumber(self):

        scaledDecayNumber = 0.0
        self.TotalSecondsMeasuring = 0.0
        decayRateSeconds = self.lambdaDays * days2seconds

        for lmfile in self.parsedListModeFiles:
            timeStamps = lmfile.time

            dTime = timeStamps[-1] - timeStamps[0]
            if dTime < (5 * 60 * 1000):
                continue

            # Detect time slots where there is recording (no significant drops)
            numBins = int(dTime / 1000.0 / 60)

            mass, edges = np.histogram(timeStamps, numBins)

            massBinary = (medfilt(mass, 11) != 0).astype(int)
            massdiff = np.array([1] + list(np.diff(massBinary)) + [-1])

            starts = np.where(massdiff > 0)[0]
            stops = np.where(massdiff < 0)[0]

            assert len(starts) == len(stops)

            # Now link this back to which DateTimes we have been actively measuring
            startTimesMS = edges[starts]
            stopTimesMS = edges[stops]  # relate this to referenceTime

            startTimes = np.datetime64(
                lmfile.fileStartDate
            ) + startTimesMS * np.timedelta64(1, "ms")
            stopTimes = np.datetime64(
                lmfile.fileStartDate
            ) + stopTimesMS * np.timedelta64(1, "ms")

            startTimeSecondsSinceReference = (
                startTimes - np.datetime64(self.referenceTime)
            ).astype("timedelta64[s]")

            stopTimeSecondsSinceReference = (
                stopTimes - np.datetime64(self.referenceTime)
            ).astype("timedelta64[s]")

            # Calculate Weighing fraction
            for iBlock in range(len(startTimes)):
                scaledDecayNumber += np.exp(
                    -decayRateSeconds
                    * startTimeSecondsSinceReference[iBlock].astype(float)
                ) - np.exp(
                    -decayRateSeconds
                    * stopTimeSecondsSinceReference[iBlock].astype(float)
                )
                self.TotalSecondsMeasuring += stopTimeSecondsSinceReference[
                    iBlock
                ].astype(float) - startTimeSecondsSinceReference[iBlock].astype(float)

        return scaledDecayNumber

    def updateDecayCorrectionScaling(self):

        scaledDecayNumber = self.getScaledDecayNumber()

        measurementToActivityFactor = (self.lambdaDays * days2seconds) / (
            self.cameraSensitivity * scaledDecayNumber
        )

        self.decayCorrectionFactor = measurementToActivityFactor

    def generatePixelatedImage(
        self,
        imageScalingFactor: int | None = None,
        decayCorrect=False,
        minClusterSize=None,
    ) -> sitk.Image:

        if imageScalingFactor is None:
            imageScalingFactor = 1

        self.imageSizeFactor = imageScalingFactor

        tmp = []
        for lmfile in self.parsedListModeFiles:

            tmp.append(
                np.histogram2d(
                    lmfile.rowCentroids,
                    lmfile.columnCentroids,
                    bins=(
                        int(lmfile.numRows * self.imageSizeFactor),
                        int(lmfile.numColumns * self.imageSizeFactor),
                    ),
                    range=[[0, lmfile.numRows], [0, lmfile.numColumns]],
                )[0]
            )

        img = np.sum(np.array(tmp), 0).astype(float)

        if decayCorrect:
            img *= self.decayCorrectionFactor

        # construct SITK Image
        returnImg = sitk.GetImageFromArray(img)
        returnImg.SetSpacing([i / imageScalingFactor * 1e3 for i in self.imageSpacing])
        # returnImg.SetSpacing([i * 1e2 for i in returnImg.GetSpacing()])
        returnImg.SetOrigin(
            [i / 2.0 for i in returnImg.GetSpacing()]
        )  # Move origin half a pixel so that different resoltuions match
        return returnImg
