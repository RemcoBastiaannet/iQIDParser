
from iQIDParser.constants import *
import os
from os.path import join as pjoin
from datetime import datetime, timedelta

import numpy as np


class iQIDListModeFile:

    time = np.ndarray | None
    rowCentroids: np.ndarray | None = None
    columnCentroids: np.ndarray | None = None
    clusterSize: np.ndarray | None = None
    clusterIntensity: np.ndarray | None = None
    acquisitionLength = None
    endOfAcq = datetime(2023, 1, 1)
    fileStartDate = datetime(2023, 1, 1)
    filteredImages: None | np.ndarray = None
    imageSize: None | int = None

    numRows: int = 1
    numColumns: int = 1

    def __init__(self, fFile: str):
        self.readFile(fFile)



    def __readCompressed(self, fFile):
        with open(fFile, "rb") as f:
            data = f.read()

        startHdr = 0
        stopHdr = 100 * sizeOfInt

        hdr = data[startHdr:stopHdr]
        body = data[stopHdr:]

        if len(body) == 0:
            return

        npHdr = np.frombuffer(hdr, dtype=int)
        self.numRows, self.numColumns = npHdr[2], npHdr[1]

        maxIX = int(len(body) / sizeOfDouble) * sizeOfDouble
        npBody = np.frombuffer(body[:maxIX], dtype=np.double)
        npBody = npBody[: int(np.floor(npBody.shape[0] / 14)) * 14]

        npBody = npBody.reshape((-1, 14))

        if npBody.shape[0] <= 1:
            return

        self.frameNumber = npBody[:, 0]
        self.time = npBody[:, 1] #elapsed time in ms

        self.clusterIntensity = npBody[:, 2]
        self.clusterSize = npBody[:, 3]
        self.rowCentroids = npBody[:, 4]
        self.columnCentroids = npBody[:, 5]
        self.eccentricity = npBody[:, 9]

        self.acquisitionLength = (self.time[-1] - self.time[0]) * 1e-3  # seconds

        # EndOfAcq

        self.endOfAcq = self.fileStartDate + timedelta(seconds=self.acquisitionLength)

    def __readCropped(self, fFile):

        with open(fFile, "rb") as f:
            data = f.read()

        startHdr = 0
        stopHdr = 100 * sizeOfInt

        hdr = data[startHdr:stopHdr]
        body = data[stopHdr:]

        if len(body) == 0:
            return

        npHdr = np.frombuffer(hdr, dtype=int)
        self.numRows, self.numColumns = npHdr[2], npHdr[1]
        self.clusterRadius = npHdr[20]
        imageSize = (2 * self.clusterRadius + 1) ** 2
        self.imageSize = imageSize

        maxIX = int(len(body) / sizeOfInt) * sizeOfInt
        npBody = np.frombuffer(body[:maxIX], dtype=np.int32)
        npBody = npBody[: int(np.floor(npBody.shape[0] / 14)) * 14]

        imageBlockSize = (
            2 * imageSize + CROPPED_LISTMODE_NUMBER_OF_FOOTER_ELEMENTS_AFTER_IMAGE
        )  # each block is filtered and unfiltered image + footer data
        nImages = int(
            len(npBody) / imageBlockSize
        )  # This contains filtered and unfiltered images
        maxIX = nImages * imageBlockSize
        npBody = npBody[:maxIX]  # trim off any partially-saved data

        imageData = npBody.reshape((-1, imageBlockSize))

        self.filteredImages = imageData[:, imageSize : 2 * imageSize]
        self.rawImages = imageData[:, :imageSize]
        footerInfo = imageData[:, 2 * imageSize :]

        self.rowCentroids = footerInfo[:, 1]
        self.columnCentroids = footerInfo[:, 2]
        self.totalSignalAboveThreshold = footerInfo[:, 3]
        self.rawClusterSignalInCroppedSum = footerInfo[:, 4]
        self.filtClusterSignalInCroppedSum = footerInfo[:, 5]

        self.time = footerInfo[:, 6] #elapsed time since start of acquisition in ms
        self.clusterSize = footerInfo[:, 7]

        self.acquisitionLength = (self.time[-1] - self.time[0]) * 1e-3  # seconds

        # EndOfAcq
        self.endOfAcq = self.fileStartDate + timedelta(seconds=self.acquisitionLength)

    def readFile(self, fFile: str):

        locFileName = os.path.basename(fFile)
        locDate, locTime = locFileName.split("_")[1:3]
        self.fileStartDate = datetime.strptime(
            locDate + "__" + locTime, "%Y-%m-%d__%Hh%Mm%Ss"
        )

        if "Compressed" in fFile:
            self.__readCompressed(fFile)
        elif "Cropped" in fFile:
            self.__readCropped(fFile)

   