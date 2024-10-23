import math


#######################
# Some constants
#######################
sizeOfInt = 4  # bytes
sizeOfDouble = 8  # bytes

alphaCameraPixSize = 1 / 18.85  # mm/pix

lambdaAc = math.log(2) / 9.92  # days
lambdaAcSeconds = math.log(2) / (9.92 *24*60*60)
days2seconds = 1.0 / (24 * 60 * 60.0)

cameraSensitivity = 0.49

CROPPED_LISTMODE_NUMBER_OF_FOOTER_ELEMENTS_AFTER_IMAGE = 8