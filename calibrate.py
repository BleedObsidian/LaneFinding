# Copyright (C) 2020 Jesse Prescott
#
# A script to take in various images containing chessboards
# and create a calibration of the intrinsics of the given
# camera.
#
# Author: Jesse Prescott

import numpy as np
import cv2
import glob
import os

# The number of inner corners of the chessboard.
chessboardSize = (9, 6)

# Termination Criteria used for Sub Pixel Calibration.
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Create a list of all the chessboard image file locations.
imageFolder = input("Please enter the folder location containing chessboard images: ")
images = glob.glob(imageFolder + '/*.jpg')

# Prepare object points.
objp = np.zeros((chessboardSize[0] * chessboardSize[1],3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# For each chessboard image.
for imageFile in images:
    
    # Load the image and convert to greyscale.
    image = cv2.imread(imageFile)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Attempt to locate chessboard corners within image.
    ret, corners = cv2.findChessboardCorners(gray, chessboardSize, None)

    # If a chessboard was found.
    if ret == True:

        # Add new row of points.
        objpoints.append(objp)

        # Calculate sub pixel corner location.
        cornersSubPixel = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(cornersSubPixel)

        # Show the image to the user and wait for them to press any key.
        image = cv2.drawChessboardCorners(image, chessboardSize, cornersSubPixel, ret)
        cv2.imshow('Chessboard Calibration', image)
        cv2.waitKey(0)

# Close any image windows still open.
cv2.destroyAllWindows()

# Calibrate the camera.
ret, cameraMatrix, distortionCoefficients, rotationVectors, translationVectors = \
    cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Save calibration to file.
if os.path.exists("calibration.npy"):
  os.remove("calibration.npy")
with open('calibration.npy', 'wb+') as calibrationFile:
    np.savez(calibrationFile, cameraMatrix=cameraMatrix, distortionCoefficients=distortionCoefficients)