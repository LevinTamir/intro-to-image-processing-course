#%% Image Segmentation
# Common to all
import os
os.chdir("C:\\image_processing_course")

#%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import the necessary packages specific to Computer vision
import cv2

# fix random seed for reproducibility
seed_value = 123; np.random.seed(seed_value); #set_random_seed(seed_value) # Keras uses its source of randomness regardless Theano or TensorFlow.In addition, TensorFlow has its own random number generator that must also be seeded by calling the set_random_seed() function immediately after the NumPy random number generator:

exec(open(os.path.abspath('image_common_utils.py')).read())

#%% Color Quantization: reduce number of colors in an image OR group colurs in few clusters

# Read and process
my_image_color = cv2.imread('./image_data/balloon.jpg')
show_image([my_image_color], row_plot = 1)

#To do KMean, convert/reshape the image to 'number of pixels' x BGR
my_image_color_BGR = np.float32(my_image_color.reshape((-1,3)))
my_image_color_BGR
my_image_color_BGR.shape # 300*225 -> my_image_color.shape

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
clusters_count = 3
ret, clusters_label, clusters_centers = cv2.kmeans(my_image_color_BGR, clusters_count, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8
clusters_centers = np.uint8(clusters_centers)
#Row side is for cluster 1,2,3

#Make original image with center color
my_image_clusters_color = clusters_centers[clusters_label.flatten()]

#reshape it back to the shape of original image
my_image_clusters_color = my_image_clusters_color.reshape((my_image_color.shape))
show_image([my_image_color, my_image_clusters_color], row_plot = 1)

#CW: Take different image and practice
#%% Watershed segmentation approach

# Read, process, treshold
img1 = cv2.imread('./image_data/water_coins.jpg')
gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the region of unknown with zero
markers[unknown==255] = 0

#apply watershed
markers = cv2.watershed(img1,markers)
img1[markers == -1] = [255,0,0]
show_image([img1, markers], row_plot = 1)

#%% Image Segmentation using Otsu Thresholding

from skimage.filters import threshold_otsu

#Read and show image
sample_image = cv2.imread('./image_data/elephant.png')
img2 = cv2.cvtColor(sample_image,cv2.COLOR_BGR2RGB)

plt.axis('off')
plt.imshow(img2)

#Apply Otsu Thresholding on Image
img2_gray=cv2.cvtColor(img2,cv2.COLOR_RGB2GRAY)

thresh = threshold_otsu(img2_gray)
img_otsu  = img2_gray < thresh

plt.imshow(img_otsu)

#segmentation
def filter_image(image, mask):

    r = image[:,:,0] * mask
    g = image[:,:,1] * mask
    b = image[:,:,2] * mask

    return np.dstack([r,g,b])

filtered = filter_image(img2, img_otsu)

plt.axis('off')
plt.imshow(filtered)
#%% Region Growth Segmentation Approach with Interactive Seed Selection

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Global list to store seed points
seed_points = []

def on_mouse_click(event, x, y, flags, param):
    """Mouse callback function to capture left-clicks and store the points."""
    if event == cv2.EVENT_LBUTTONDOWN:
        seed_points.append((y, x))  # Append (y, x) since OpenCV uses (x, y) but numpy uses (y, x)

def region_grow(img, seeds, threshold):
    """Region growth algorithm."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    for seed in seeds:
        stack = [seed]  # Stack for seed points
        while stack:
            x, y = stack.pop()
            if not mask[x, y] and abs(img[x, y] - img[seed]) < threshold:
                mask[x, y] = True
                # Neighboring pixels
                neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
                for nx, ny in neighbors:
                    if 0 <= nx < h and 0 <= ny < w:
                        stack.append((nx, ny))
    # Apply mask to the image
    segmented_img = np.zeros_like(img)
    segmented_img[mask] = img[mask]
    return segmented_img

def main(img_path, threshold):
    global seed_points
    img = cv2.imread(img_path, 0).astype('float32')  # Load the image in grayscale
    
    # Display image and set mouse callback
    #cv2.namedWindow('Image')
    cv2.namedWindow('Image', cv2.WINDOW_AUTOSIZE)
    #cv2.imshow('Image', img)
    cv2.setMouseCallback('Image', on_mouse_click)
    cv2.imshow('Image', my_image_color)
    cv2.waitKey(0)  # Wait for key press
    cv2.destroyAllWindows()

    # Apply region growing with the selected seed points
    if seed_points:
        segmented_img = region_grow(img, seed_points, threshold)
        
        # Display the original and segmented images
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(segmented_img, cmap='gray')
        plt.title('Segmented Image')
        plt.axis('off')

        plt.show()
    else:
        print("No seed points selected.")

# Parameters
img_path = './image_data/balloon.jpg'  # Specify your image path
threshold = 20.0  # Threshold for region growth

if __name__ == "__main__":
    main(img_path, threshold)


