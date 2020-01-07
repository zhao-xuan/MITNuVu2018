import cv2
import numpy as np

assert float(cv2.__version__.rsplit('.', 1)[0]) >= 3, 'OpenCV version 3 or newer required.'

"""K = np.array([[  689.21,     0.  ,  1295.56],
              [    0.  ,   690.48,   942.17],
              [    0.  ,     0.  ,     1.  ]])"""
K = np.array([[  404.41/1.3,     0.  ,  486/2],
              [    0.  ,   302.89/1.3,   364/2],
              [    0.  ,     0.  ,     1.  ]])
# zero distortion coefficients work well for this image
D = np.array([0., 0., 0., 0.])

# use Knew to scale the output
Knew = K.copy()
Knew[(0,1), (0,1)] = 0.4 * Knew[(0,1), (0,1)]


"""warped = imread('warped.png')
unwarped = imread('unwarped.png')
cv2.initCameraMatrix2D()"""

img = cv2.imread('Capture4.png')
cv2.imshow('test', img)
cv2.waitKey(2000)
img_undistorted = cv2.fisheye.undistortImage(img, K, D=D, Knew=Knew)
h, w, _ = img_undistorted.shape
img_undistorted = img_undistorted[(h/3-30):(h/3-50)+h/2, w/4:w/4+w/2]
cv2.imwrite('undistorted.jpg', img_undistorted)
cv2.imshow('undistorted', img_undistorted)
cv2.waitKey(3000)
