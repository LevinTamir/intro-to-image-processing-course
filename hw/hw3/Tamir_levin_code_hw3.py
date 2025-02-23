#%%
#!/usr/bin/env python
# coding: utf-8

"""
HW3 - Introduction to Image Processing Course

Tamir Levin

"""
# %% Imports
import sys
import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.image import imread, imsave
import pandas as pd
import numpy as np
import tensorflow as tf
import skimage as ski
from skimage.io import imread as ski_imread
from skimage import img_as_float, exposure
from skimage.filters import threshold_otsu
from PIL import Image
import six.moves.urllib as urllib
import tarfile
import zipfile

# Additional imports for object detection and segmentation
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append("model\\models-master\\research\\")
from object_detection.utils import label_map_util, visualization_utils as vis_util
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score

# %% Global Output Folder
OUTPUT_FOLDER = "image_output"

# %% Library Versions
def print_library_versions():
    """Print versions of key libraries."""
    print('pandas version:', pd.__version__)
    print('numpy version:', np.__version__)
    print('tensorflow version:', tf.__version__)
    print('scikit-image version:', ski.__version__)
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    print(f"Current Conda environment: '{conda_env}'.")


# %% Utility Functions for Image Display and Saving
def display_multiple_images(*images, titles=None):
    """
    Display multiple images side by side.

    Parameters:
        images (list): List of image arrays.
        titles (list, optional): Titles for each image.
    """
    images = [img_as_float(img) for img in images]
    if titles is None:
        titles = [''] * len(images)
    vmin = min(map(np.min, images))
    vmax = max(map(np.max, images))
    ncols = len(images)
    height = 5
    width = height * ncols
    fig, axes = plt.subplots(1, ncols, figsize=(width, height))
    for ax, img, label in zip(axes.ravel(), images, titles):
        ax.imshow(img, vmin=vmin, vmax=vmax)
        ax.set_title(label)
        ax.axis('off')
    plt.show()


def display_image(image, title="Image"):
    """
    Display a single image with a title.

    Parameters:
        image (array): Image array.
        title (str): Title for the display window.
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()


def save_image(image, filename, output_folder=OUTPUT_FOLDER):
    """
    Save an image to the output folder.
    
    Parameters:
        image (array): Image array.
        filename (str): Filename to use for saving.
        output_folder (str): Folder where the image will be saved.
    """
    os.makedirs(output_folder, exist_ok=True)
    save_path = os.path.join(output_folder, filename)
    cv2.imwrite(save_path, image)
    print(f"Saved output to: {save_path}")


# %% Face Detection and Recognition Functions
def detect_face_region(input_img):
    """
    Detect a face in the input image and return the cropped region along with its bounding box.

    Parameters:
        input_img (array): The input image (BGR format).

    Returns:
        Tuple of (face_region, rect) where face_region is the cropped grayscale face 
        and rect is the bounding box (x, y, w, h). Returns (-1, -1) if no face is found.
    """
    gray_image = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('./model/haarcascades/haarcascade_frontalface_alt2.xml')
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.2, minNeighbors=5)
    if len(faces) == 0:
        return -1, -1
    (x, y, w, h) = faces[0]
    return gray_image[y:y+h, x:x+w], faces[0]


def detect_and_draw_eyes(input_img):
    """
    Detect eyes in the image and draw rectangles around them.

    Parameters:
        input_img (array): The input image (BGR format).

    Returns:
        Image with detected eyes marked.
    """
    img_copy = input_img.copy()
    gray_image = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.equalizeHist(gray_image)
    eye_cascade = cv2.CascadeClassifier('./model/haarcascades/haarcascade_eye.xml')
    eyes_rects = eye_cascade.detectMultiScale(
        gray_image,
        scaleFactor=1.05,
        minNeighbors=7,
        minSize=(20, 20),
        maxSize=(100, 100)
    )
    for (x, y, w, h) in eyes_rects:
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return img_copy


def load_face_training_data(training_folder_path):
    """
    Load and prepare face training data.

    Parameters:
        training_folder_path (str): Path to the folder with training images organized by label.

    Returns:
        detected_faces (list): List of processed face images.
        face_labels (list): Corresponding labels for each face.
    """
    detected_faces = []
    face_labels = []
    training_image_dirs = os.listdir(training_folder_path)
    
    for dir_name in training_image_dirs:
        label = int(dir_name)  # Assumes folder names are numeric labels
        training_image_path = os.path.join(training_folder_path, dir_name)
        training_images_names = os.listdir(training_image_path)
        
        for image_name in training_images_names:
            image_path = os.path.join(training_image_path, image_name)
            image = cv2.imread(image_path)
            face, rect = detect_face_region(image)
            if face is not -1:
                resized_face = cv2.resize(face, (121, 121), interpolation=cv2.INTER_AREA)
                detected_faces.append(resized_face)
                face_labels.append(label)
    return detected_faces, face_labels


def draw_bounding_rectangle(image, rect):
    """Draw a bounding rectangle on the image."""
    (x, y, w, h) = rect
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)


def draw_label_text(image, text, x, y):
    """Draw label text on the image."""
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)


def predict_face(input_img, recognizer, tags):
    """
    Predict the label for the detected face in the image.

    Parameters:
        input_img (array): Input image (BGR format).
        recognizer: Trained face recognizer.
        tags (list): List of label names.

    Returns:
        Tuple of (annotated image, predicted label as text).
    """
    detected_face, rect = detect_face_region(input_img)
    if detected_face is -1:
        return input_img, "Unknown"
    
    resized_face = cv2.resize(detected_face, (121, 121), interpolation=cv2.INTER_AREA)
    label = recognizer.predict(resized_face)
    label_text = tags[label[0]]
    draw_bounding_rectangle(input_img, rect)
    draw_label_text(input_img, label_text, rect[0], rect[1] - 5)
    return input_img, label_text


def process_image_file(image_path, recognizer, tags, output_folder=OUTPUT_FOLDER):
    """
    Process a single image file for face recognition, display the result,
    and save the output in the common output folder.

    Parameters:
        image_path (str): Path to the test image.
        recognizer: Trained face recognizer.
        tags (list): List of label names.
        output_folder (str): Folder to save the output image.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not open image file {image_path}")
        return
    predicted_image, _ = predict_face(image, recognizer, tags)
    base_name = os.path.basename(image_path)
    save_image(predicted_image, f"Predicted_{base_name}", output_folder)
    display_image(predicted_image, title="Predicted Face")


def process_video_file(video_path, recognizer, tags, output_folder=OUTPUT_FOLDER):
    """
    Process a video file for face recognition in real-time, display the output,
    and save the processed video to the common output folder.
    
    The processed video is saved with a filename prefixed by "Processed_".
    
    Parameters:
        video_path (str): Path to the video file.
        recognizer: Trained face recognizer.
        tags (list): List of label names.
        output_folder (str): Folder to save the processed video.
    """
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Retrieve video properties
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    width  = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    
    # Prepare output video file path
    base_name = os.path.basename(video_path)
    output_video_path = os.path.join(output_folder, f"Processed_{base_name}")
    os.makedirs(output_folder, exist_ok=True)
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    print(f"Processing video. Output will be saved to: {output_video_path}")
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        predicted_frame, _ = predict_face(frame, recognizer, tags)
        video_writer.write(predicted_frame)
        cv2.imshow("Video Face Recognition", predicted_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    video_writer.release()
    cv2.destroyAllWindows()
    print("Video processing complete.")


# %% Object Detection Functions
MODEL_NAMES = [
    'faster_rcnn_inception_v2_coco_2018_01_28',
    'ssd_inception_v2_coco_2017_11_17'
]
MODEL_DIR = './model/'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
PATH_TO_LABELS = os.path.join('model', 'models-master', 'research', 'object_detection', 'data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


def download_object_detection_model(model_name):
    """
    Download and extract an object detection model.

    Parameters:
        model_name (str): Name of the model.

    Returns:
        Path to the frozen inference graph.
    """
    model_tar = model_name + '.tar.gz'
    model_path = tf.keras.utils.get_file(
        fname=model_tar,
        origin=DOWNLOAD_BASE + model_tar,
        untar=True,
        cache_dir=MODEL_DIR,
        cache_subdir='downloaded'
    )
    return os.path.join(model_path, 'frozen_inference_graph.pb')


def load_tensorflow_model(model_path):
    """
    Load a TensorFlow model from a frozen graph.

    Parameters:
        model_path (str): Path to the frozen inference graph.
    
    Returns:
        TensorFlow detection graph.
    """
    detection_graph = tf.compat.v1.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph


def detect_objects(image, detection_graph):
    """
    Perform object detection on an image using the specified TensorFlow graph.

    Parameters:
        image (array): Input image (BGR format).
        detection_graph: Loaded TensorFlow detection graph.

    Returns:
        Image with detection boxes drawn.
    """
    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            image_np_expanded = np.expand_dims(image, axis=0)
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded}
            )
            vis_util.visualize_boxes_and_labels_on_image_array(
                image,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=4
            )
    return image


# %% IoU Functions (Integrated into Object Detection and Segmentation)
def compute_iou(box_a, box_b, epsilon=1e-5):
    """
    Compute the Intersection over Union (IoU) between two bounding boxes.

    Box format: (x1, y1, x2, y2). The function reorders coordinates
    to ensure proper computation even if the points are not ordered.
    """
    # Reorder box_a coordinates
    a_x1 = min(box_a[0], box_a[2])
    a_y1 = min(box_a[1], box_a[3])
    a_x2 = max(box_a[0], box_a[2])
    a_y2 = max(box_a[1], box_a[3])
    
    # Reorder box_b coordinates
    b_x1 = min(box_b[0], box_b[2])
    b_y1 = min(box_b[1], box_b[3])
    b_x2 = max(box_b[0], box_b[2])
    b_y2 = max(box_b[1], box_b[3])
    
    # Compute the coordinates of the intersection rectangle
    inter_x1 = max(a_x1, b_x1)
    inter_y1 = max(a_y1, b_y1)
    inter_x2 = min(a_x2, b_x2)
    inter_y2 = min(a_y2, b_y2)
    
    # Compute the width and height of the intersection rectangle
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    area_overlap = inter_width * inter_height

    # Compute areas of both bounding boxes
    area_a = (a_x2 - a_x1) * (a_y2 - a_y1)
    area_b = (b_x2 - b_x1) * (b_y2 - b_y1)
    
    # Compute union area and IoU
    area_union = area_a + area_b - area_overlap
    iou = area_overlap / (area_union + epsilon)
    
    return iou


def interactive_iou_annotation(image):
    """
    Launch an interactive tool to annotate two bounding boxes and compute their IoU.
    
    Use the mouse to draw rectangles; press 'r' to reset and 'q' to quit.
    """
    clone = image.copy()
    rectangles = []

    def draw_rectangle(event, x, y, flags, param):
        nonlocal rectangles, clone
        if event == cv2.EVENT_LBUTTONDOWN:
            rectangles.append((x, y, x, y))
        elif event == cv2.EVENT_LBUTTONUP:
            rectangles[-1] = rectangles[-1][:2] + (x, y)
            if len(rectangles) % 2 == 0:
                color = (0, 0, 255)
                cv2.rectangle(clone, rectangles[-1][:2], rectangles[-1][2:], color, 2)
                iou_value = compute_iou(rectangles[-2], rectangles[-1])
                print(f"Intersection over Union (IoU): {iou_value:.2f}")
                cv2.putText(clone, f'IoU = {iou_value:.3f}', (rectangles[-1][0]+5, rectangles[-1][1]+15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.imshow("Image", clone)

    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", draw_rectangle)

    while True:
        cv2.imshow("Image", clone)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            clone = image.copy()
            rectangles = []
        elif key == ord("q"):
            break
    cv2.destroyAllWindows()
    return clone


def print_model_label_map(model_name, category_index):
    """
    Print the label map for a given model.

    Parameters:
        model_name (str): The name of the model.
        category_index (dict): The category index dictionary.
    """
    print(f"\nLabel map for {model_name}:")
    for idx, info in category_index.items():
        print(f"{idx}: {info['name']}")


def compare_object_detection_models(image_filename, model1_graph, model2_graph, output_folder=OUTPUT_FOLDER):
    """
    Compare object detection outputs from two models on a single image.
    Saves the comparison image and then runs IoU annotation on it.

    Parameters:
        image_filename (str): Name of the test image file (located in './image_data/').
        model1_graph: TensorFlow graph for the first model.
        model2_graph: TensorFlow graph for the second model.
        output_folder (str): Folder to save the output images.
    """
    image_path = os.path.join('image_data', image_filename)
    image_orig = cv2.imread(image_path)
    if image_orig is None:
        raise FileNotFoundError(f"Image '{image_filename}' not found in 'image_data/' folder")
    
    # Print label maps for both models
    print_model_label_map("Faster R-CNN", category_index)
    print_model_label_map("SSD Inception", category_index)
    
    orig_height, orig_width = image_orig.shape[:2]
    output1 = detect_objects(image_orig.copy(), model1_graph)
    output2 = detect_objects(image_orig.copy(), model2_graph)
    
    output1_rgb = cv2.cvtColor(output1, cv2.COLOR_BGR2RGB)
    output2_rgb = cv2.cvtColor(output2, cv2.COLOR_BGR2RGB)
    
    output1_resized = cv2.resize(output1_rgb, (orig_width, orig_height))
    output2_resized = cv2.resize(output2_rgb, (orig_width, orig_height))
    
    fig, axes = plt.subplots(1, 2, figsize=(orig_width/100, orig_height/100))
    axes[0].imshow(output1_resized)
    axes[0].set_title("Faster RCNN Detection", fontsize=12)
    axes[0].axis('off')
    axes[1].imshow(output2_resized)
    axes[1].set_title("SSD Inception Detection", fontsize=12)
    axes[1].axis('off')
    plt.tight_layout()
    
    os.makedirs(output_folder, exist_ok=True)
    image_name_no_ext = os.path.splitext(image_filename)[0]
    base_filename = "Detection_Comparison_" + image_name_no_ext
    save_path = os.path.join(output_folder, f"{base_filename}.jpg")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    
    # --- IoU Annotation integrated into Object Detection ---
    image_iou = cv2.imread(save_path)
    if image_iou is not None:
        scale = 0.8
        image_iou = cv2.resize(image_iou, None, fx=scale, fy=scale)
        annotated_image = interactive_iou_annotation(image_iou)
        save_image(annotated_image, base_filename + '_IoU.jpg', output_folder)
    else:
        print("Could not load image for IoU annotation.")


# %% Segmentation Functions
def watershed_segmentation(image_path, output_path=os.path.join(OUTPUT_FOLDER, 'watershed.jpg')):
    """
    Perform watershed segmentation on the image.

    Parameters:
        image_path (str): Path to the input image.
        output_path (str): Path to save the output segmentation result.
    """
    img = cv2.imread(image_path)
    img = cv2.resize(img, None, fx=1, fy=1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [0, 0, 255]
    
    # Save the segmentation result first
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Watershed Segmentation')
    plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2)
    plt.imshow(markers)
    plt.title('Markers')
    plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.show()
    print(f"Segmentation result saved to: {output_path}")
    
    # Launch interactive IoU annotation on the saved segmentation image
    image_seg = cv2.imread(output_path)
    if image_seg is not None:
        annotated_image = interactive_iou_annotation(image_seg)
        base_filename = os.path.splitext(os.path.basename(output_path))[0]
        annotated_path = os.path.join(os.path.dirname(output_path), base_filename + '_IoU.jpg')
        cv2.imwrite(annotated_path, annotated_image)
        print(f"Annotated segmentation (with IoU) saved to: {annotated_path}")
    else:
        print("Could not load image for IoU annotation.")


def kmeans_segmentation(image_path, clusters_count=8, output_path=os.path.join(OUTPUT_FOLDER, 'K-means_8cluster.png')):
    """
    Perform K-means segmentation on the image and evaluate clustering metrics.

    Parameters:
        image_path (str): Path to the input image.
        clusters_count (int): Number of clusters to form.
        output_path (str): Path to save the segmented image.
    """
    image = cv2.imread(image_path, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixel_vals = image.reshape((-1, 3))
    pixel_vals = np.float32(pixel_vals)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, clusters_label, clusters_centers = cv2.kmeans(pixel_vals, clusters_count, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    clusters_centers = np.uint8(clusters_centers)
    segmented_image = clusters_centers[clusters_label.flatten()]
    segmented_image = segmented_image.reshape(image.shape)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image', fontsize=15)
    plt.subplot(1, 2, 2)
    plt.imshow(segmented_image)
    plt.title(f'Clustered Image - {clusters_count} Clusters', fontsize=15)
    plt.savefig(output_path, dpi=200)
    plt.show()
    print(f"Segmentation result saved to: {output_path}")
    
    # Launch interactive IoU annotation on the saved k-means segmentation image
    image_seg = cv2.imread(output_path)
    if image_seg is not None:
        annotated_image = interactive_iou_annotation(image_seg)
        base_filename = os.path.splitext(os.path.basename(output_path))[0]
        annotated_path = os.path.join(os.path.dirname(output_path), base_filename + '_IoU.png')
        cv2.imwrite(annotated_path, annotated_image)
        print(f"Annotated segmentation (with IoU) saved to: {annotated_path}")
    else:
        print("Could not load image for IoU annotation.")


def otsu_threshold_segmentation(image_path, output_path=os.path.join(OUTPUT_FOLDER, 'Otsu_Threshold.jpg')):
    """
    Perform image segmentation using Otsu Thresholding.

    Parameters:
        image_path (str): Path to the input image.
        output_path (str): Path to save the segmentation result.
    """
    def filter_image(image, mask):
        r = image[:, :, 0] * mask
        g = image[:, :, 1] * mask
        b = image[:, :, 2] * mask
        return np.dstack([r, g, b])
    
    sample_image = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(img_rgb)
    plt.axis('off')
    
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    thresh = threshold_otsu(img_gray)
    img_otsu = img_gray < thresh
    
    plt.subplot(1, 3, 2)
    plt.title(f'Otsu Threshold: {thresh:.2f}')
    plt.imshow(img_otsu, cmap='gray')
    plt.axis('off')
    
    filtered = filter_image(img_rgb, img_otsu)
    plt.subplot(1, 3, 3)
    plt.title('Filtered Image')
    plt.imshow(filtered)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.show()
    print(f"Segmentation result saved to: {output_path}")
    
    # Launch interactive IoU annotation on the saved Otsu segmentation image
    image_seg = cv2.imread(output_path)
    if image_seg is not None:
        annotated_image = interactive_iou_annotation(image_seg)
        base_filename = os.path.splitext(os.path.basename(output_path))[0]
        annotated_path = os.path.join(os.path.dirname(output_path), base_filename + '_IoU.jpg')
        cv2.imwrite(annotated_path, annotated_image)
        print(f"Annotated segmentation (with IoU) saved to: {annotated_path}")
    else:
        print("Could not load image for IoU annotation.")


# %% Main Functions
def main_face_detection_recognition():
    """Run tasks for face detection and recognition and save outputs to the common folder."""
    print_library_versions()
    
    training_data_folder = './image_data/train-data'
    tags = ['Tamir']  # Example tag; adjust as needed
    
    print("Preparing face training data...")
    faces, labels = load_face_training_data(training_data_folder)
    
    # Initialize and train the face recognizer
    recognizer = cv2.face.EigenFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    
    # Process an example image for face recognition
    test_image_path = "./image_data/test-data/0/test1.jpg"
    process_image_file(test_image_path, recognizer, tags, output_folder=OUTPUT_FOLDER)
    
    # Detect and display (and save) eyes on an example image
    img = cv2.imread(test_image_path)
    eyes_image = detect_and_draw_eyes(img)
    save_image(eyes_image, "Detected_Eyes.jpg", OUTPUT_FOLDER)
    display_image(eyes_image, title="Detected Eyes")
    
    # Process an example video for face recognition and save the processed video
    video_path = "./image_data/test-data/0/test4.mp4"  # Update path as needed
    process_video_file(video_path, recognizer, tags, output_folder=OUTPUT_FOLDER)


def main_object_detection():
    """Run tasks for object detection (with integrated IoU annotation) and save outputs to the common folder."""
    print("Downloading and loading object detection models...")
    model_paths = {name: download_object_detection_model(name) for name in MODEL_NAMES}
    model1_graph = load_tensorflow_model(model_paths['faster_rcnn_inception_v2_coco_2018_01_28'])
    model2_graph = load_tensorflow_model(model_paths['ssd_inception_v2_coco_2017_11_17'])
    
    # Print label maps for both models
    print_model_label_map("Faster R-CNN", category_index)
    print_model_label_map("SSD Inception", category_index)
    
    image_filename = 'multiple_objects3.jpg'
    compare_object_detection_models(image_filename, model1_graph, model2_graph, output_folder=OUTPUT_FOLDER)


def main_segmentation():
    """Run tasks for segmentation demos and save outputs to the common output folder."""
    watershed_segmentation('./image_data/water_coins.jpg', output_path=os.path.join(OUTPUT_FOLDER, 'watershed.jpg'))
    kmeans_segmentation('./image_data/animal.jpg', clusters_count=8, output_path=os.path.join(OUTPUT_FOLDER, 'K-means_8cluster.png'))
    otsu_threshold_segmentation('./image_data/water_coins.jpg', output_path=os.path.join(OUTPUT_FOLDER, 'Otsu_Threshold.jpg'))


# %% Run Selected Main Function(s)
if __name__ == "__main__":
    
    # Run Face Detection and Recognition tasks (including video processing):
     main_face_detection_recognition()
    
    # Run Object Detection tasks (with IoU annotation):
     main_object_detection()
    
    # Run Segmentation tasks (with integrated IoU annotation):
     main_segmentation()
