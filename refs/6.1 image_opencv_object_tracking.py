#Object Tracking: Concepts on ppt
#source ref: https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/

import os
os.chdir("D:\\image_processing_course")

#%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cv2 # pip install opencv-contrib-python
cv2.__version__
from random import randint

# Some standard settings
plt.rcParams['figure.figsize'] = (13, 9) #(16.0, 12.0)
plt.style.use('ggplot')

#Set PANDAS to show all columns in DataFrame
pd.set_option('display.max_columns', None)
#Set PANDAS to show all rows in DataFrame
pd.set_option('display.max_rows', None)
pd.set_option('display.precision', 2)

# fix random seed for reproducibility
seed_value = 123; np.random.seed(seed_value); #set_random_seed(seed_value) # Keras uses its source of randomness regardless Theano or TensorFlow.In addition, TensorFlow has its own random number generator that must also be seeded by calling the set_random_seed() function immediately after the NumPy random number generator:

exec(open(os.path.abspath('image_common_utils.py')).read())

#%%  Object tracking
#pip install opencv-contrib-python
#tracker = cv2.TrackerBoosting_create() # not accurate in any scenario
#tracker = cv2.TrackerMIL_create() # not accurate in any scenario
#tracker = cv2.TrackerKCF_create() # not accurate in any scenario
#tracker = cv2.TrackerCSRT_create() # every N frame although less accurate
#tracker = cv2.TrackerTLD_create() # not accurate in any scenario
#tracker = cv2.TrackerMedianFlow_create() # every consecutive frame
#tracker = cv2.TrackerMOSSE_create() # every consecutive frame

#%% Helper functions
def select_objects(window_name, frame):
    
    ## Select boxes
    bboxes = []
    colors = [] 
    
    # OpenCV's selectROI function doesn't work for selecting multiple objects in Python
    # So we will call this function in a loop till we are done selecting all objects
    while True:
      # draw bounding boxes over objects
      # selectROI's default behaviour is to draw box starting from the center
      # when fromCenter is set to false, you can draw box starting from top left corner
      bbox = cv2.selectROI(window_name, frame, fromCenter=False)
      bboxes.append(bbox)
      colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
      print("Press q to quit selecting boxes and start tracking")
      print("Press any other key to select next object")
      k = cv2.waitKey(0) & 0xFF
      if k == 113 or k == 27:  # q or esc is pressed
        break
    
    print('Selected bounding boxes {}'.format(bboxes))    
    return bboxes, colors
#end of select_objects
    
def get_initialized_tracker(bboxes, frame):
    
    # Create MultiTracker object if more than one baox has been selected
    multiTracker = cv2.legacy.MultiTracker_create()
    
    # Initialize MultiTracker 
    for bbox in bboxes:
        #tracker = cv2.TrackerCSRT_create() # every N frame although less accurate
        #multiTracker.add(tracker, frame, bbox)
        multiTracker.add(cv2.legacy.TrackerKCF_create(), frame, bbox)
      
    return multiTracker
#end of get_initialized_tracker   

#%%Object tracking: objects on Video

window_name = 'tracking'

#Read the video data
capture_video = cv2.VideoCapture('./image_data/Shopping People Commerce Mall Many Crowd Walking   Free Stock video footage   YouTube.mp4')
#capture_video = cv2.VideoCapture('./image_data/Test video for Object Detection -- TRIDE.mp4')

#Read first frame to select the area
ret, frame = capture_video.read()

#Select the person or area to track
bboxes, colors = select_objects(window_name, frame)

# Initialize tracker with first frame and bounding box
tracker = get_initialized_tracker(bboxes, frame)

frame_rate = int(capture_video.get(cv2.CAP_PROP_FPS))

while(capture_video.isOpened()):
    ret, frame = capture_video.read()

    if ret==True:
        frame_seq = int(capture_video.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_seq % frame_rate >= 0: # for total use CAP_PROP_FRAME_COUNT
            # Update tracker
            tracker_ret, boxes = tracker.update(frame)
              # Draw bounding box
            if tracker_ret:
                # draw tracked objects
                for i, newbox in enumerate(boxes):
                    p1 = (int(newbox[0]), int(newbox[1]))
                    p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                    cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
            else :
                # Tracking failure
                cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
            
            #Display
            cv2.imshow(window_name, frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 113 or k == 27: # ord('q') -> 113, Esc -> 27
                break
    else:
        break

# Release everything if job is finished
capture_video.release()
cv2.destroyAllWindows()

del(tracker, capture_video, ret, frame, boxes, tracker_ret, frame_rate)
