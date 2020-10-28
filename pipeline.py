# Copyright (C) 2020 Jesse Prescott
#
# This file contains all of the parts of the pipeline
# to detect the lane lines of a given image.
#
# Author: Jesse Prescott

import numpy as np
import cv2

def undistort(image, calibrationFile):
    """
        Takes the given image and calibration file to return
        an undistorted image.
    """

    # Extract camera calibration data fro given file.
    calibrationData = np.load(calibrationFile)
    cameraMatrix = calibrationData['cameraMatrix']
    distortionCoefficients = calibrationData['distortionCoefficients']

    # Return undistorted image.
    return cv2.undistort(image, cameraMatrix, distortionCoefficients)

def extract_features(image, s_thresh=(170, 255), sx_thresh=(20, 100)):
    """
        Takes the given image and creates a binary image
        highlighting possible lane line features.
    """

    # Convert to HLS color space and seperate the lightness and saturation channels.
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    # Apply the sobel operator in the x direction.
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Apply a threshold to the sobel output to create a binary image.
    sx_binary = np.zeros_like(scaled_sobel)
    sx_binary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Apply a threshold to the saturation channel.
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Stack each channel
    combined_binary = np.zeros_like(sx_binary)
    combined_binary[(sx_binary == 1) | (s_binary == 1)] = 255

    # Coloured binary for debuging.
    #color_binary = np.dstack(( np.zeros_like(sx_binary), sx_binary, s_binary)) * 255
    #cv2.imwrite("color_binary.jpg", color_binary)
    #cv2.imwrite("binary.jpg", combined_binary)

    # Return final binary image.
    return combined_binary

def perspective_transform(image):
    """
        Apply a perspective transform to make the image
        appear as though it's from a bird's eye view.
    """

    