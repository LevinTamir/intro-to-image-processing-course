import os
os.chdir("D:\\image_processing_course")

#%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cv2

# fix random seed for reproducibility
seed_value = 123; np.random.seed(seed_value); #set_random_seed(seed_value) # Keras uses its source of randomness regardless Theano or TensorFlow.In addition, TensorFlow has its own random number generator that must also be seeded by calling the set_random_seed() function immediately after the NumPy random number generator:

exec(open(os.path.abspath('image_common_utils.py')).read())

#%% Vehicle detection approaches

#Completed object detection using supervised learning approaches.

#Unsupervised object detection
#Count the number of objects
#Find the relative size of the objects
#Find the relative distance between the objects
#Above comes under unsupervised (without using any labeled) way of object detection

#For Vehicle: approach
#Frame differencing
#See miscellaneous.xlsx -> tab 'image_diff'

#%%  Vehicle detection: Test data prepration

#Load video
capture_video = cv2.VideoCapture('./image_data/A3 A-road Traffic UK HD - rush hour - British Highway traffic May 2017 short.mp4')

while(capture_video.isOpened()):
    ret, frame = capture_video.read()
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if ret==True:
        if capture_video.get(cv2.CAP_PROP_POS_FRAMES) % 30 >= 0: # for total use CAP_PROP_FRAME_COUNT
            cv2.imshow('frame',frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 113 or k == 27: # ord('q') -> 113, Esc -> 27
                break

        #Save few frame for learning
        frame_seq = int(capture_video.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_seq in [30, 31, 60, 90] :
            file_name = './image_output/vehicle_' + str(frame_seq) + '.png'
            cv2.imwrite(file_name,frame)

    else:
        break

# Release everything if job is finished
capture_video.release()
cv2.destroyAllWindows()

#%%Vehicle detection: Frame (say image) difference

#Function to detect moving object across frames
def get_marked_frame(previous_frame, current_frame):
    # convert the frames to grayscale
    previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    diff_image = cv2.absdiff(current_frame_gray, previous_frame_gray)
    #cv2_show_fullscr('image', diff_image)

    # perform image thresholding
    ret, im_th = cv2.threshold(diff_image, 40, 255, cv2.THRESH_BINARY)# for INV, background will be white
    #cv2_show_fullscr('image', im_th)

    #Pause for few minutes and decide what need to be done with small white dots
    #Strategy: Remove, Ignore now

    # Run erosion and dilation
    kernel = np.ones((5,5),np.uint8)
    my_image_dilation = cv2.dilate(im_th, kernel,iterations = 3)
    #cv2_show_fullscr('image', my_image_dilation)

    #Find contours: Now let’s find the contours in the detection zone of the above frame:
    contours, hierarchy = cv2.findContours(my_image_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # cv2.RETR_CCOMP

    #The first check is whether the top-left y-coordinate of the contour should be >= 80 (I am including one more check, x-coordinate <= 200). The other check is that the area of the contour should be >= 25. You can find the contour area with the help of the cv2.contourArea( ) function.
    current_frame_marked = current_frame.copy(); valid_contours = []
    for i,contour in enumerate(contours):
        if cv2.contourArea(contour) >= 500:
            x,y,w,h = cv2.boundingRect(contour)
            #cv2.rectangle(current_frame_marked,(x,y),(x+w,y+h),(255, 0, 0),3)
            valid_contours.append(contour)

    # count of discovered contours
    len(valid_contours)

    current_frame_marked = cv2.drawContours(current_frame_marked, valid_contours, -1, (255, 0, 0), 2)
    #cv2_show_fullscr('image', current_frame_marked)

    return current_frame_marked
#end of get_marked_frame

#Read test images
previous_frame = cv2.imread('./image_output/vehicle_30.png',1)
current_frame = cv2.imread('./image_output/vehicle_31.png',1)

#Call above functions
current_frame_marked = get_marked_frame(previous_frame, current_frame)
show_image([cv2.cvtColor(previous_frame, cv2.COLOR_BGR2RGB), cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB), cv2.cvtColor(current_frame_marked, cv2.COLOR_BGR2RGB)], row_plot = 1)
cv2_show_fullscr('image', current_frame_marked)

#%%Vehicle detection: in video

#Now let use above function in video.
capture_video = cv2.VideoCapture('./image_data/A3 A-road Traffic UK HD - rush hour - British Highway traffic May 2017 short.mp4')
previous_frame = None

while(capture_video.isOpened()):
    ret, frame = capture_video.read()

    if ret==True:
        if previous_frame is None:
            previous_frame = frame
            continue
        elif capture_video.get(cv2.CAP_PROP_POS_FRAMES) % 30 >= 0: # for total use CAP_PROP_FRAME_COUNT

            current_frame_marked = get_marked_frame(previous_frame, frame)

            cv2.imshow('frame',current_frame_marked)

            k = cv2.waitKey(1) & 0xFF
            previous_frame = frame

            if k == 113 or k == 27: # ord('q') -> 113, Esc -> 27
                break
    else:
        break

# Release everything if job is finished
capture_video.release()
cv2.destroyAllWindows()

#CW: Practice with any other similar video and share your experince

#%% Lane Detection: Concepts
#The Concept of Lane Detection: https://en.wikipedia.org/wiki/Lane

# Various approaches
# Supervised: Deep learning model
# Manual using OpenCV : Detect white-colored lane markings

#%% Lane Detection: Test data prepration

#Load video
capture_video = cv2.VideoCapture('./image_data/lane_video.mp4')

while(capture_video.isOpened()):
    ret, frame = capture_video.read()
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if ret==True:
        if capture_video.get(cv2.CAP_PROP_POS_FRAMES) % 30 >= 0: # for total use CAP_PROP_FRAME_COUNT
            cv2.imshow('frame',frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 113 or k == 27: # ord('q') -> 113, Esc -> 27
                break

        #Save two frame for learning
        frame_seq = int(capture_video.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_seq  in [30, 31, 60, 90] :
            file_name = './image_output/lane_' + str(frame_seq) + '.png'
            cv2.imwrite(file_name,frame)

    else:
        break

# Release everything if job is finished
capture_video.release()
cv2.destroyAllWindows()

#%% Lane Detection: on current Frame (say image)

#Function to detect lane in current frames using Hough transformation
def draw_lane_line_using_hough(current_frame):

    #grayscale because we only need the luminance channel for detecting edges
    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
    #cv2_show_fullscr('image', current_frame_gray)

    # Applies a 5x5 gaussian blur with deviation of 0 to frame - not mandatory since Canny will do this for us
    current_frame_gray_blur = cv2.GaussianBlur(current_frame_gray, (5, 5), 0)
    #cv2_show_fullscr('image', current_frame_gray_blur)

    # Applies Canny edge detector with minVal of 50 and maxVal of 150
    current_frame_gray_blur_canny = cv2.Canny(current_frame_gray_blur, 50, 150)
    #cv2_show_fullscr('image', current_frame_gray_blur_canny)

    #Only manual step: Get area of interest
    #As per file '1.1 image_sklearn_basics' : [rows, columns, channels] -> [H , W , C]

    H = current_frame_gray_blur_canny.shape[0]; W = current_frame_gray_blur_canny.shape[1]

    #show_image([cv2.cvtColor(current_frame_gray_blur_canny, cv2.COLOR_BGR2RGB)], row_plot = 1)

    # Creates a triangular polygon for the mask defined by three (x, y) coordinates
    polygons = np.array([[(0, H), (W, H), (400, 300)]]) # As pe manual view of image

    # Creates an image filled with zero intensities with the same dimensions as the frame
    current_frame_black_mask = np.zeros_like(current_frame_gray_blur_canny)

    # Allows the mask to be filled with values of 1 and the other areas to be filled with values of 0
    current_frame_black_mask = cv2.fillPoly(current_frame_black_mask, polygons, 255)

    # A bitwise and operation between the mask and frame keeps only the triangular area of the frame
    segment_of_interest = cv2.bitwise_and(current_frame_gray_blur_canny, current_frame_black_mask)
    #cv2_show_fullscr('image', segment_of_interest)

    #Hough Transform is a technique to detect any shape that can be represented mathematically.
    #For example, it can detect shapes like rectangles, circles, triangles, or lines.
    #https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html
    #https://en.wikipedia.org/wiki/Hough_transform
    #Hough Line Transformation: https://en.wikipedia.org/wiki/Hough_transform

    hough = cv2.HoughLinesP(segment_of_interest, 2, np.pi / 180, 100, np.array([]), minLineLength = 100, maxLineGap = 50)

    if hough is not None:
        for i in range(0, len(hough)):
            l = hough[i][0]
            cv2.line(current_frame, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)

    #cv2_show_fullscr('image', current_frame)

    return(current_frame)
#end of draw_lane_line_using_hough

#Read test images
current_frame = cv2.imread('./image_output/lane_30.png',1)
cv2_show_fullscr('image', current_frame)

current_frame = draw_lane_line_using_hough(current_frame)
cv2_show_fullscr('image', current_frame)

#%% Lane Detection: in video

#Now let use above function in video.
capture_video = cv2.VideoCapture('./image_data/lane_video.mp4')

while(capture_video.isOpened()):
    ret, current_frame = capture_video.read()

    if ret==True:
        if capture_video.get(cv2.CAP_PROP_POS_FRAMES) % 30 >= 0: # for total use CAP_PROP_FRAME_COUNT

            current_frame_marked = draw_lane_line_using_hough(current_frame)

            cv2.imshow('frame',current_frame_marked)

            k = cv2.waitKey(1) & 0xFF
            if k == 113 or k == 27: # ord('q') -> 113, Esc -> 27
                break
    else:
        break

# Release everything if job is finished
capture_video.release()
cv2.destroyAllWindows()

#CW: How to make generic . i.e to identify any lanes
