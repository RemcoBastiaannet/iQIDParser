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

cameraSensitivity = .49*4*.725*1.24015234#.04* 3.06700385 * .5#.49 * 4 #-> lambdaAc???? 1 ==> 1/4 particles detected

CROPPED_LISTMODE_NUMBER_OF_FOOTER_ELEMENTS_AFTER_IMAGE = 8