# Copyright (C) 2020 Jesse Prescott
#
# A script to find the current lane of an ego vehicle with a front facing
# dashcam.
#
# Author: Jesse Prescott

from pipeline import *
import cv2

# Ask the user for the location of the video file.
videoFile = "videos/project_video.mp4" #input("Please provide video file path: ")
videoCapture = cv2.VideoCapture(videoFile)
videoCapture.set(1, 1)
newFrame, image = videoCapture.read()

# Create output video object.
video = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'MP4V'), videoCapture.get(cv2.CAP_PROP_FPS), (image.shape[1], image.shape[0]))

# Ask the user for the calibration file.
calibrationFile = "calibration.npy" #input("Please provide the calibration file path: ")

# Warn user about the length of the process.
print("Processing video, this may take a while...")

# While we still have new images to process.
count = 1
while newFrame:

    # First undistort the image with the given calibration.
    image = undistort(image, calibrationFile)

    # Create a binary image containing possible lane line features.
    binary = extract_features(image)

    # Transform the perspective to a bird's eye view.
    birdsEye = perspective_transform(binary)

    # Get two polynomials for the lane lines.
    final_image = find_lane_lines(birdsEye)

    # Add text to the image.
    cv2.putText(final_image, 'Frame Number: ' + str(count), (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)

    # Add image to our output video.
    video.write(final_image)

    # Load the next image to process.
    newFrame, image = videoCapture.read()

    # Increment count.
    count += 1

# Finalalise video.
video.release()