# Copyright (C) 2020 Jesse Prescott
#
# This file contains all of the parts of the pipeline
# to detect the lane lines of a given image.
#
# Author: Jesse Prescott

#from roipoly import RoiPoly
from matplotlib import pyplot as plt
import numpy as np
import cv2

# Used to store the inverse matrix between functions.
inverseTransformMatrix = None
originalImage = None
usedSlidingWindow = False
previous_poly_left = None
previous_poly_right = None

def undistort(image, calibrationFile):
    """
        Takes the given image and calibration file to return
        an undistorted image.
    """

    global originalImage

    # Extract camera calibration data fro given file.
    calibrationData = np.load(calibrationFile)
    cameraMatrix = calibrationData['cameraMatrix']
    distortionCoefficients = calibrationData['distortionCoefficients']

    # Return undistorted image.
    originalImage = cv2.undistort(image, cameraMatrix, distortionCoefficients)
    return originalImage

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

    # Stack each binary image together.
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

    global inverseTransformMatrix

    # Uses a matplotlib library to allow the user to easily select
    # a trapazoid ROI, this only need to be done once.
    #plt.imshow(image[:,:,::-1])
    #roi = RoiPoly(color='r')
    #print(roi.x)
    #print(roi.y)

    # Trapazoid as selected by the user.
    tl = [297, 667]
    tr = [1023, 665]
    br = [679, 443]
    bl = [605, 444]
    trapazoid = np.float32([tl, tr, br, bl])

    # Calculation of destination points for trapazoid.
    height, width = image.shape
    x_window_width = 350
    imageVerticies = np.float32([[0+x_window_width, height], [width-x_window_width, height], [width-x_window_width, 0], [0+x_window_width, 0]])

    # Calculate perspective transform.
    transformMatrix = cv2.getPerspectiveTransform(trapazoid, imageVerticies)
    inverseTransformMatrix = cv2.getPerspectiveTransform(imageVerticies, trapazoid)

    # Apply perspective transform.
    image = cv2.warpPerspective(image, transformMatrix, (width, height))

    return image

def find_lane_lines(image):
    """
        Find the lane lines using a sliding window histogram
        and return two polynomials that represent the two
        lane lines. A binary top-down image must be given.
    """

    global inverseTransformMatrix, originalImage, usedSlidingWindow, previous_poly_left, previous_poly_right

    # Copy binary image with all three channels so we
    # can make a debug image later.
    debug_image = np.dstack((image, image, image))

    # Male a list of all the x and y positions of non-black
    # pixels.
    nonzero = image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Use sliding window for first frame.
    if not usedSlidingWindow:

        # Get the bottom quarter of the image and create
        # a histogram.
        image_bottom = image[image.shape[0]//2:,:]
        histogram = np.sum(image_bottom, axis=0)
        #plt.plot(histogram)
        #plt.show()

        # Find the highest peaks on the left and righ half of the histogram.
        histogram_midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:histogram_midpoint])
        rightx_base = np.argmax(histogram[histogram_midpoint:]) + histogram_midpoint

        # Sliding window parameters.
        nwindows = 9
        window_width = 180
        minimum_pixels = 50

        # Calculate the window height.
        window_height = np.int(image.shape[0]//nwindows)

        # Current window positions.
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Arrays to store left and right pixel indicies.
        left_lane_pixels = []
        right_lane_pixels = []

        # For every sliding window row.
        for window in range(nwindows):

            # Calculate window boundaries for both left and right lanes.
            win_y_low = image.shape[0] - (window+1)*window_height
            win_y_high = image.shape[0] - window*window_height
            win_xleft_low = leftx_current - window_width
            win_xleft_high = leftx_current + window_width
            win_xright_low = rightx_current - window_width
            win_xright_high = rightx_current + window_width

            # Draw the windows on the visualization image
            cv2.rectangle(debug_image,(win_xleft_low,win_y_low),
            (win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(debug_image,(win_xright_low,win_y_low),
            (win_xright_high,win_y_high),(0,255,0), 2)

            # Identify the non-black pixels in x and y within the window.
            good_left_pixels = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_pixels = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

            # Append these pixels to our lists.
            left_lane_pixels.append(good_left_pixels)
            right_lane_pixels.append(good_right_pixels)

            # If we have found more than the minimum number of pixels
            # in our window, recenter the next one on their average
            # position.
            if len(good_left_pixels) > minimum_pixels:
                leftx_current = np.int(np.mean(nonzerox[good_left_pixels]))
            if len(good_right_pixels) > minimum_pixels:
                rightx_current = np.int(np.mean(nonzerox[good_right_pixels]))

        # Concatenate our arrays to make one long array.
        try:
            left_lane_pixels = np.concatenate(left_lane_pixels)
            right_lane_pixels = np.concatenate(right_lane_pixels)
        except ValueError:
            # TODO: Implement edge case later.
            return None

        # Extract left and right lane pixel positions
        leftx = nonzerox[left_lane_pixels]
        lefty = nonzeroy[left_lane_pixels] 
        rightx = nonzerox[right_lane_pixels]
        righty = nonzeroy[right_lane_pixels]

        # Fit a second order polynomial through our pixels.
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Dont use sliding windows again.
        usedSlidingWindow = True
        previous_poly_left = left_fit
        previous_poly_right = right_fit

    else:

        # Find activated pixels within a certain margin of our previous polynomials.
        margin = 20
        left_lane_pixels = ((nonzerox > (previous_poly_left[0]*(nonzeroy**2) + previous_poly_left[1]*nonzeroy + 
                previous_poly_left[2] - margin)) & (nonzerox < (previous_poly_left[0]*(nonzeroy**2) + 
                previous_poly_left[1]*nonzeroy + previous_poly_left[2] + margin)))
        right_lane_pixels = ((nonzerox > (previous_poly_right[0]*(nonzeroy**2) + previous_poly_right[1]*nonzeroy + 
                        previous_poly_right[2] - margin)) & (nonzerox < (previous_poly_right[0]*(nonzeroy**2) + 
                        previous_poly_right[1]*nonzeroy + previous_poly_right[2] + margin)))

        # Extract left and right lane pixel positions
        leftx = nonzerox[left_lane_pixels]
        lefty = nonzeroy[left_lane_pixels] 
        rightx = nonzerox[right_lane_pixels]
        righty = nonzeroy[right_lane_pixels]

        # Fit a second order polynomial through our pixels.
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        previous_poly_left = left_fit
        previous_poly_right = right_fit

    # Use or polynomial and some linear points to draw
    # it and check the fit on our debug image.
    ploty = np.linspace(0, image.shape[0]-1, image.shape[0])
    try:
        left_fit_x = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fit_x = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        print("Failed to fit polynomial.")

    # Draw our polynomials and colour the pixels used to calculate
    # them.
    debug_image[lefty, leftx] = [255, 0, 0]
    debug_image[righty, rightx] = [0, 0, 255]
    plt.plot(left_fit_x, ploty, color='yellow')
    plt.plot(right_fit_x, ploty, color='yellow')
    #plt.imshow(debug_image)
    #plt.show()

    # Create a new blank image for us to draw the lane
    # that will be used to overlay on top of the original
    # image.
    mask_image = np.zeros_like(image).astype(np.uint8)
    mask_image_color = np.dstack((mask_image, mask_image, mask_image))

    # Change the datatype of our points so we can use cv2.fillPoly()
    points_left = np.array([np.transpose(np.vstack([left_fit_x, ploty]))])
    points_right = np.array([np.flipud(np.transpose(np.vstack([right_fit_x, ploty])))])
    points = np.hstack((points_left, points_right))

    # Fill the region created by our polynomials points.
    cv2.fillPoly(mask_image_color, np.int_([points]), (0, 255, 0))

    # Transform the masked image back from bird's eye view.
    mask_image_color = cv2.warpPerspective(mask_image_color, inverseTransformMatrix, (image.shape[1], image.shape[0]))

    # Combine the masked image and the original image.
    final_image = cv2.addWeighted(originalImage, 1, mask_image_color, 0.3, 0)

    # Calculate curvature.
    ym_per_pix = 400/720 # The meters per pixel in y dimension
    xm_per_pix = 3.7/700 # The meters per pixel in x dimension
    y_eval = np.max(ploty)
    left_curvature = ((1 + (2*left_fit[0]*y_eval*ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curvature = ((1 + (2*right_fit[0]*y_eval*ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    curvature = int((left_curvature + right_curvature) / 2)

    # Add the curvature as text to our image.
    if curvature <= 5000:
        cv2.putText(final_image, 'Curvature: ' + str(curvature) + "m", (10, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(final_image, 'Curvature: None', (10, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)

    # Calculate the cars offset from the lane centre.
    left_point = left_fit_x[-1]
    right_point = right_fit_x[-1]
    lane_midpoint = (left_point + right_point) / 2
    offset = round(((originalImage.shape[1]/2) - lane_midpoint) * xm_per_pix, 2)
    cv2.putText(final_image, 'Ego Car Offset: ' + str(offset) + "m", (10, 60), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)

    # Return our final image.
    return final_image
