#Object detection: Concepts on ppt
import os
os.chdir("D:\\image_processing_course")

#%matplotlib inline
import os
import sys
import numpy as np
import pandas as pd
from scipy.spatial import distance
import matplotlib.pyplot as plt

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

#%%  TensorFlow Object detection
# Source: https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/index.html
# Source: https://github.com/tensorflow/models/tree/master/research/object_detection


import six.moves.urllib as urllib
import tarfile
import zipfile
import tensorflow as tf
tf.__version__ # '2.10.0'

#from collections import defaultdict
#from io import StringIO
#from PIL import Image

# Since working folder is different then current working directory and hence adding the path
#sys.path.append("model\\models-master\\models-master\\research\\")
sys.path.append("model\\models-master\\research\\")
#from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
# It may throw some warning. Ignore warning
from object_detection.utils import visualization_utils as vis_util

# import the necessary packages specific to Computer vision
import cv2

# What model to download.
# Models can bee found here: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
MODEL_NAME = 'faster_rcnn_inception_v2_coco_2018_01_28' # 'ssd_inception_v2_coco_2017_11_17' 'faster_rcnn_inception_v2_coco_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
MODEL_FILEPATH = './model/' + MODEL_FILE
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# List of the strings that is used to add correct label for each box.
#PATH_TO_LABELS = os.path.join('model\\models-master\\models-master\\research\\object_detection\\data', 'mscoco_label_map.pbtxt')
PATH_TO_LABELS = os.path.join('model\\models-master\\research\\object_detection\\data', 'mscoco_label_map.pbtxt')

# Number of classes to detect
NUM_CLASSES = 90 # max number of classes for given model

WINDOW_NAME = 'object detection'

# Download Model: Run only once per model file
model_dir = tf.keras.utils.get_file(fname=MODEL_NAME, origin=DOWNLOAD_BASE + MODEL_FILE, untar=True, cache_dir ='./model/', cache_subdir= 'downloaded')

# Load a (frozen) Tensorflow model into memory.
model = tf.saved_model.load(model_dir + '/saved_model')
model = model.signatures['serving_default']

# Loading label map: indices to category names
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
categories # see the objects that will be predicted
category_index = label_map_util.create_category_index(categories)

#%% Helper functions
#Single helper function to read image, predict the objects
image_file_name = './model/models-master/research/object_detection/test_images/image2.jpg' #'./image_output/social_distancing_75.png'
def get_numpy_image_and_predictions(model, image_file_name):
    #Default when incoming value is image.
    image_np = image_file_name
    if isinstance(image_file_name, str):
        image_np = cv2.imread(image_file_name, cv2.IMREAD_COLOR)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    # Detection
    #Check the model’s input signature (it expects a batch of 3-color images of type int8):
    model.inputs # The first one is batch or count of images
    model.output_dtypes

    image_np = np.asarray(image_np)
    
    # The input needs to be a tensor
    image_np_tensor = tf.convert_to_tensor(image_np)
    #image_np_tensor.shape
    # The model expects a batch of image_nps, so add an axis with `tf.newaxis`.
    image_np_tensor = image_np_tensor[tf.newaxis,...]

    # Run inference
    pred_dict = model(image_np_tensor)

    # All outputs are batches tensors. Convert to numpy arrays, and take index [0] to remove the
    #batch dimension. We're only interested in the first num_detections.
    num_detections = int(pred_dict.pop('num_detections'))
    pred_dict = {key:value[0, :num_detections].numpy() for key,value in pred_dict.items()}
    pred_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    pred_dict['detection_classes'] = pred_dict['detection_classes'].astype(np.int64)

    del(image_np_tensor, num_detections)

    return image_np, pred_dict
#end of get_numpy_image_and_predictions

#It calculates distance between two images and if within 100 (say) then mark those in red
def social_distance_on_single_image(image_np, pred_dict):

    #Convert the prediction to data frame for easy processing. We need class and cordinates
    df_pred = pd.concat([pd.DataFrame({'detection_scores': pred_dict['detection_scores'] , 'detection_classes': pred_dict['detection_classes']}), pd.DataFrame(pred_dict['detection_boxes'])],axis=1)
    df_pred.head(2)
    df_pred.columns = ['detection_scores', 'detection_classes', 'y1', 'x1', 'y2', 'x2']

    #Since we need person with high score only and hence get person class only
    df_pred = df_pred[(df_pred['detection_scores'] > 0.5) & (df_pred['detection_classes'] == 1)]
    df_pred.reset_index(drop = True, inplace = True)
    if df_pred.shape[0] == 0:
        print('No person detected in image')
        return image_np

    #The co-ordinates are normalized. Bring to original image shape for drawing
    df_pred['y1'] = df_pred['y1'] * image_np.shape[0]
    df_pred['y2'] = df_pred['y2'] * image_np.shape[0]

    df_pred['x1'] = df_pred['x1'] * image_np.shape[1]
    df_pred['x2'] = df_pred['x2'] * image_np.shape[1]

    #Calculate centroids. Take mid point and bottom (leg) of the person
    centroids = [(int((df_pred.loc[row_num,'x1'] + df_pred.loc[row_num,'x2'])/2), int(df_pred.loc[row_num,'y2'])) for row_num in range(0, df_pred.shape[0])]

    # Use standard algoritums to get distances
    np_dist_matrix = distance.cdist(centroids, centroids, 'euclidean')

    index_to_highlight = set()# to avoid duplicates
    # loop over the upper triangular of the distance matrix
    for i in range(0, np_dist_matrix.shape[0]):
    	for j in range(i + 1, np_dist_matrix.shape[1]):
    		if np_dist_matrix[i, j] < 100: # Hardcoded for learning purpose
    			index_to_highlight.add(i)
    			index_to_highlight.add(j)

    #Draw rectangles with green or red colors
    for row_num in range(0, df_pred.shape[0]):
        color = (0, 255, 0) # green

        if row_num in index_to_highlight:
            color = (0, 0, 255) # red
            
        #image_np = cv2.rectangle(image_np, (df_pred.loc[row_num,'x1'], df_pred.loc[row_num,'y1']), (df_pred.loc[row_num,'x2'], df_pred.loc[row_num,'y2']), color, 2)
        x1 = int(df_pred.loc[row_num,'x1'])
        y1 = int(df_pred.loc[row_num,'y1'])
        x2 = int(df_pred.loc[row_num,'x2'])
        y2 = int(df_pred.loc[row_num,'y2'])
        image_np = cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)


    text = "Currect frame: Social Distancing within 100 unit: {}".format(len(index_to_highlight))
    cv2.putText(image_np, text, (15, image_np.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

    #cv2_show_fullscr('social_distance_on_single_image', image_np)

    del(df_pred, centroids, np_dist_matrix, index_to_highlight)
    return image_np
#end of social_distance_on_single_image

#Single function to read image, predict the objects and display
#image_file_name = './model/models-master/research/object_detection/test_images/image2.jpg'; cv2_waitKeyTime = 0; social_distance = False
def predict_and_display_for_single_image(model, image_file_name, cv2_waitKeyTime = 0, social_distance = False):

    #helper function to read image, predict the objects
    image_np, pred_dict = get_numpy_image_and_predictions(model, image_file_name)

    #Draw above information on image.
    if social_distance:
        #Note: Incoming is in RGB and convert to BGR for seemless cv2 operations
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        image_np = social_distance_on_single_image(image_np, pred_dict)
        #image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        #cv2_show_fullscr('image', image_np)
    else:
        vis_util.visualize_boxes_and_labels_on_image_array(image_np, pred_dict['detection_boxes'], pred_dict['detection_classes'], pred_dict['detection_scores'], category_index, instance_masks = pred_dict.get('detection_masks_reframed', None),  use_normalized_coordinates=True, line_thickness=8)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Display output
    cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)# for full screen

    cv2.imshow(WINDOW_NAME, image_np); cv2_waitKey = cv2.waitKey(cv2_waitKeyTime)

    #Useful for images only when wait time is infinite (0)
    if cv2_waitKeyTime == 0:
        cv2.destroyAllWindows()

    del(pred_dict)
    return cv2_waitKey
#end of predict_and_display_for_single_image

#%% Object detection on image
# Read image
image_file_names = ['./model/models-master/research/object_detection/test_images/image2.jpg', './image_data/elephant.png', './image_data/horse_text_digit_2.png']
for image_file_name in image_file_names:
    print('Processing file ' + image_file_name)
    predict_and_display_for_single_image(model, image_file_name)

#CW: Try with some other images. Preferably with objects mentioned in 'categories'
#CW: Do object detection in any video with objects mentioned in 'categories'
#%%Social Distancing: Test data prepration

capture_video = cv2.VideoCapture('./image_data/Shopping People Commerce Mall Many Crowd Walking   Free Stock video footage   YouTube.mp4')


# for full screen
#cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
#cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

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
        frame_rate = int(capture_video.get(cv2.CAP_PROP_FPS))

        if frame_seq  in [frame_rate, 2*frame_rate, 3*frame_rate, 4*frame_rate] :
            file_name = './image_output/social_distancing_' + str(frame_seq) + '.png'
            cv2.imwrite(file_name,frame)

    else:
        break

# Release everything if job is finished
capture_video.release()
cv2.destroyAllWindows()

#%% Social Distancing on image

# Read image
image_file_names = ['./image_output/social_distancing_25.png', './image_output/social_distancing_50.png', './image_output/social_distancing_75.png', './image_output/social_distancing_100.png']
for image_file_name in image_file_names:
    print('Processing file ' + image_file_name)
    predict_and_display_for_single_image(model, image_file_name, social_distance = True)

#Loks like './image_output/social_distancing_75.png' is good image for practice
image_file_name = './image_output/social_distancing_75.png'
predict_and_display_for_single_image(model, image_file_name, social_distance = True)

#%%Social Distancing: on video

capture_video = cv2.VideoCapture('./image_data/Shopping People Commerce Mall Many Crowd Walking   Free Stock video footage   YouTube.mp4')

# for full screen
#cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
#cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

while(capture_video.isOpened()):
    ret, frame = capture_video.read()
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if ret==True:
        frame_rate = int(capture_video.get(cv2.CAP_PROP_FPS))
        if capture_video.get(cv2.CAP_PROP_POS_FRAMES) % frame_rate == 0: # for total use CAP_PROP_FRAME_COUNT
            #cv2.imshow('frame',frame)
            k = predict_and_display_for_single_image(model, frame, cv2_waitKeyTime = 1, social_distance = True)
#            k = cv2.waitKey(1) & 0xFF
            if k == 113 or k == 27: # ord('q') -> 113, Esc -> 27
                break
    else:
        break

# Release everything if job is finished
capture_video.release()
cv2.destroyAllWindows()

