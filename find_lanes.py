# Copyright (C) 2020 Jesse Prescott
#
# A script to find a lane in the given video.
#
# Author: Jesse Prescott

from pipeline import *
import cv2

# Ask the user for the location of the video file.
videoFile = "videos/project_video.mp4" #input("Please provide video file path: ")
videoCapture = cv2.VideoCapture(videoFile)
newFrame, image = videoCapture.read()

# Ask the user for the calibration file.
calibrationFile = "calibration.npy" #input("Please provide the calibration file path: ")

# While we still have new images to process.
while newFrame:

    # First undistort the image with the given calibration.
    image = undistort(image, calibrationFile)

    # Create a binary image containing possible lane line features.
    binary = extract_features(image)

    
    break

    newFrame, image = videoCapture.read()