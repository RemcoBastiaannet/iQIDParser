import os
from types import SimpleNamespace

import numpy as np
import SimpleITK as sitk

from iQIDParser.iQIDParser import iQIDParser, pjoin

import matplotlib.pyplot as plt

# load files --> Generate pixelated image to select different masks
iQID = iQIDParser(r"F:\DATA\iQID\October 2023 Calibration Check\Listmode")
# iQID = iQIDParser(r"F:\DATA\iQID\Ac225Calibration\Listmode")

alphaImg = iQID.generatePixelatedImage(imageScalingFactor=1)

# #%% Define masks
import matplotlib.pyplot as plt
import json

# %matplotlib qt
# # Load the grayscale image
# image = np.log(sitk.GetArrayFromImage(alphaImg)+1)

# # Create a figure and axis for the image
# fig, ax = plt.subplots()
# ax.imshow(image, cmap='inferno')

# # Define a list to store the polygon coordinates
# polygons = []

# # Define a function to handle mouse events
# def draw_polygon(event):
#     global polygons
#     if event.button == 1:
#         # Create a new polygon
#         polygon = []
#         polygon.append((event.xdata, event.ydata))
#         polygons.append(polygon)
#     elif event.button == 3:
#         # Complete the polygon
#         polygon = polygons[-1]
#         polygon.append((event.xdata, event.ydata))
#     elif event.button == 2:
#         # Save the polygon coordinates to a JSON file
#         with open('polygons.json', 'w') as f:
#             json.dump(polygons, f)
#         print('Polygon coordinates saved to polygons.json')
#         return

#     # Draw the polygons on the image
#     for polygon in polygons:
#         x, y = zip(*polygon)
#         ax.plot(x, y, 'r')

#     # Update the plot
#     plt.draw()

# # Set the mouse event callback function
# fig.canvas.mpl_connect('button_press_event', draw_polygon)

# # Show the image
# plt.show()



#%%

with open('polygons.json', 'r') as f:
    polygons = json.load(f)


#Plot decay curves
time, nDecays = iQID.decayCurve()
time = np.array(time)
nDecays = np.array(nDecays)
timeDays = time/1000/60/60/24

plt.plot(time[1::]/1000/60/60, nDecays[1:]/np.mean(nDecays[:10]))
plt.plot(time[1::]/1000/60/60, np.exp(-iQID.lambdaDays * timeDays[1:]))



#Normalize back to time of calibration

#Check error --> Tune threshold, etc.


######################
