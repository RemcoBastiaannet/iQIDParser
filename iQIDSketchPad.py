from iQIDParser import iQIDParser

import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

from scipy.optimize import curve_fit
from scipy.signal import medfilt

# %%
fiQIDData = r"Y:\September 2024 Timepoint 2\Listmode"
iQID: iQIDParser = iQIDParser(fiQIDData, listmodeType="Compressed")
img = iQID.generatePixelatedImage(decayCorrect = True)


# %%

referenceTime = iQID.referenceTime
decayRateDays = np.log(2) / 9.9
decayRateMinutes = np.log(2) / (9.9 * 24 * 60)


scaledDecayNumber = 0.
for lmfile in iQID.parsedListModeFiles:
    timeStamps = lmfile.time

    dTime = timeStamps[-1] - timeStamps[0]

    if dTime < (5 * 60 * 1000):
        continue

    #Detect time slots where there is recording (no significant drops)
    numBins = int(dTime / 1000.0 / 60)

    mass, edges = np.histogram(timeStamps, numBins)

    # plt.figure()
    # plt.plot(edges[:-1]/1000./60/60, mass)
    # plt.ylabel('Counts per minute')
    # plt.xlabel('Time [h]')
    # plt.savefig('ExampleOfDroppingData.png')

    massBinary = (medfilt(mass, 11) != 0).astype(int)
    massdiff = np.array([1] + list(np.diff(massBinary)) + [-1])

    starts = np.where(massdiff > 0)[0]
    stops = np.where(massdiff < 0)[0]

    assert len(starts) == len(stops)

    # Now link this back to which DateTimes we have been actively measuring
    startTimesMS = edges[starts]
    stopTimesMS = edges[stops]  # relate this to referenceTime

    startTimes = np.datetime64(lmfile.fileStartDate) + startTimesMS * np.timedelta64(
        1, "ms"
    )
    stopTimes = np.datetime64(lmfile.fileStartDate) + stopTimesMS * np.timedelta64(
        1, "ms"
    )

    startTimeDaysSinceReference = (startTimes - np.datetime64(referenceTime)).astype(
        "timedelta64[m]"
    )
    stopTimeDaysSinceReference = (stopTimes - np.datetime64(referenceTime)).astype(
        "timedelta64[m]"
    )

    #Calculate Weighing fraction
    for iBlock in range(len(startTimes)):
        scaledDecayNumber += np.exp(-decayRateMinutes * startTimeDaysSinceReference[iBlock].astype(int)) - np.exp(
            -decayRateMinutes * stopTimeDaysSinceReference[iBlock].astype(int)
        )


ActivityFactor = decayRateMinutes / ( iQID.cameraSensitivity *  )