import numpy as np
import cv2
from matplotlib import pyplot as plt

# Define camera matrix K
K = np.array([[  404.41,     0.  ,  486/2],
              [    0.  ,   302.89,   364/2],
              [    0.  ,     0.  ,     1.  ]])
# zero distortion coefficients work well for this image
d = np.array([2.3, 0.1, -0.1, -0.1])

# Define distortion coefficients d
#d = np.array([5.44787247e-02, 1.23043244e-01, -4.52559581e-04, 5.47011732e-03, -6.83110234e-01])
#d = np.array([-0.36824145, 0.2848545, 0.079123, 0.00064924, -0.16345661])

# Read an example image and acquire its size
img = cv2.imread("pic2.png")
h, w = img.shape[:2]

# Generate new camera matrix from parameters
newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(K, d, (w,h), 0)

# Generate look-up tables for remapping the camera image
mapx, mapy = cv2.initUndistortRectifyMap(K, d, None, newcameramatrix, (w, h), 5)

# Remap the original image to a new image
newimg = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

# Display old and new image
fig, (oldimg_ax, newimg_ax) = plt.subplots(1, 2)
oldimg_ax.imshow(img)
oldimg_ax.set_title('Original image')
newimg_ax.imshow(newimg)
newimg_ax.set_title('Unwarped image')
plt.show()
