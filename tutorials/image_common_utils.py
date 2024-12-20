#%% Get min (either of Width or Height), Max and Mean of size of image. Also get count of images
# for each tuple of dimensions
# image_dir = "./image_age/Train/"
def get_image_size_summary(image_dir):
    from keras.preprocessing import image

    # Get list of all files asuming that files are image type only
    #list_of_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    list_of_files = image.list_pictures(image_dir)
    #  arr_txt = [x for x in os.listdir() if x.endswith(".txt")]
    print("Count of image files read is ",  len(list_of_files))

    # Create a temporary DF to staore image sizes
    df = pd.DataFrame()

    # Iterate over each images and read their sizes
    for img_name in list_of_files: # [0:10] img_name = list_of_files[0]
        #img_path = os.path.join(image_dir, img_name)
        img_path = img_name
        img = image.load_img(img_path) # , target_size=(224, 224)
        df = pd.concat([df, pd.DataFrame({'W' : [img.size[0]], 'H' : [img.size[1]]})],axis=0) #top/bottom rbind
        del(img)

    # Get mIn, Max and Mean sizes
    size_min = np.min(df.min()); size_max = np.max(df.max()); size_mean = np.int(np.mean(df.mean()))
    print("Min, Max and Mean are ", size_min, size_max, size_mean)

    df = get_group_cat_many(df, df.columns.tolist())

    return({"size_min" : size_min, "size_max" : size_max, "size_mean" : size_mean,"summary" :df})

# Read the image
# W = img_summary['size_min']; H = W
def load_images(image_dir, W, H, image_array_divided_by = 1):
    from keras.preprocessing import image

    # Get list of all files asuming that files are image type only
    list_of_files = image.list_pictures(image_dir)
    print("Count of image files read is ",  len(list_of_files))

    # Iterate over each images and read their sizes
    list_images = []
    for img_name in list_of_files: #[0:10] img_name = list_of_files[0]
        img_path = img_name
        img = np.NaN
        #img = image.load_img(img_path, target_size=(W, H))
        if W > 0 and H > 0:
            img = image.load_img(img_path, target_size=(W, H))
        else:
            img = image.load_img(img_path)

        x = image.img_to_array(img)
        x /= image_array_divided_by
        list_images.append(x)
        del(img, x)

    # Stack to have one complete list as required by NN
    train = np.stack(list_images)

    # Cleaning
    del(list_images)

    # Return all required
    return({"train" : train, "list_of_files" : list_of_files})
    # end of load_images

# Description: Return 3D shape based on K Channel. Need to be made 4d before Analysis
def get_input_shape(count, img_rows, img_cols):
    if K.image_data_format() == 'channels_first':
        input_shape = (count, img_rows, img_cols)
    else: # "Count - Height-Width-Depth"
        input_shape = (img_rows, img_cols, count)

    return(input_shape)

# Description: It displays many images in one row
def imshow_all(*images, titles=None):
    images = [img_as_float(img) for img in images]

    if titles is None:
        titles = [''] * len(images)
    vmin = min(map(np.min, images))
    vmax = max(map(np.max, images))
    ncols = len(images)
    height = 5
    width = height * len(images)
    fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(width, height))
    for ax, img, label in zip(axes.ravel(), images, titles):
        ax.imshow(img, vmin=vmin, vmax=vmax)
        ax.set_title(label)

    return
    # end of imshow_all

# Description: It displays many images in many rows and column
# cmap='gray'; row_plot = 1; img= [img, camera]
def show_image(img, cmap='gray', row_plot = 1):
    if not isinstance(img, list):
        img = [img]

    if len(img) == 1:
        plt.imshow(img[0], cmap=cmap) # cm.gray
    else:
        # Get column count
        col_plot = len(img) // row_plot
        fig, axes = plt.subplots(row_plot, col_plot)
        for count in range(len(img)): # count= 1
            axes[count].imshow(img[count], cmap=cmap)
    plt.show()
    return
# end of show_image

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

#Use ImageDataGenerator and view first 10 and save 100 samples
#img_path = './image_data/elephant.png'; out_folder = './image_output/augmented/'; count_see = 10; count_save = 12
def generate_multiorientedimage_using_ImageDataGenerator_and_view(img_path,out_folder = './image_output/augmented/',count_see = 10, count_save = 1000, seed_value = 123):

    from skimage.io import imread, imsave

    # Read images
    my_color_image = imread(img_path)
    my_color_image.shape # H x W x C

    # Extend dimension (the first one is number of images)
    my_color_image = np.expand_dims(my_color_image,0) # add num_images dimension
    my_color_image.shape

    # Prepare generator object with lot of configurations
    generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                    zoom_range=0.3, # Zoom randomly by a factor of 0.3
                    rotation_range=50, # Rotate randomly by 50 degrees
                    width_shift_range=0.2, # Translate randomly horizontally by a factor of 0.2
                    height_shift_range=0.2, # Translate randomly verticallly by a factor of 0.2
                    shear_range=0.2, # Applying shear-based transformations randomly
                    horizontal_flip=True,
                    vertical_flip = True,
                    fill_mode='nearest') # Leveraging the fill_mode parameter to fill in new
                                        # pixels for images after the preceding operations

    # Generate batches of augmented images from this image
    iterators = generator.flow(my_color_image, seed = seed_value)

    # Get 10 samples of augmented images
    augmented_images = [next(iterators) for i in range(count_see)]

    #View the first 10 images and the class name
    plt.figure(figsize=(10,10))
    for i in range(count_see):
        plt.subplot(2,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        img_temp = augmented_images[i]
        x_image = np.reshape(img_temp, (img_temp.shape[1], img_temp.shape[2], img_temp.shape[3]))
        plt.imshow(x_image) # , cmap=plt.cm.binary
        #plt.xlabel(np.argmax(y[i]))
        # end of 'for'

    #Need to get fresh iterators as previous one is already travelled by 10
    iterators = generator.flow(my_color_image, seed = seed_value)

    # get file/image name
    file_name = img_path.replace('.png','')
    if file_name.rfind('/') == -1:
        file_name = 'image_augmented'
    else:
        file_name = file_name[file_name.rfind('/') + 1:]

    #Save the images
    for i in range(count_save):
        img_temp = next(iterators)
        img_temp = np.reshape(img_temp, (img_temp.shape[1], img_temp.shape[2], img_temp.shape[3]))
        imsave(out_folder + '/' + file_name + str(i) + '.png', img_temp)

# generate_multiorientedimage_using_ImageDataGenerator_and_view
#generate_multiorientedimage_using_ImageDataGenerator_and_view(img_path,out_folder,count_see, count_save)

#%% cv2 related
def cv2_show_fullscr(window_name, list_my_image_color):
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

#    if len(list_my_image_color) > 1:
#    list_my_image_color = np.hstack(list_my_image_color)

    cv2.imshow(window_name, list_my_image_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# end of cv2_show_fullscr

# Take help from http://answers.opencv.org/question/99282/does-findcontours-create-duplicates/
# Description: It drop duplicate with area overlap of 'epsilon' overlap.
# There might be small contours - suppress all of them if their area is less than 'area_suppress'
def cv2_get_unique_contours(my_image_color, contours, epsilon =0.05, area_suppress = 100):
    # create dummy image with all 0
    mask = np.zeros(my_image_color.shape,np.uint8)

    # The multiplication may become big and hence scaling to 0-1
    my_image_color_temp = my_image_color.copy().astype(np.float)/my_image_color.max().astype(np.float)

    #get unique contours
    contours_new = []
    for contour in contours: # cv2.boundingRect(contours[7])
        x,y,w,h = cv2.boundingRect(contour)
        if area_suppress < w*h and np.sum(mask[y:y+h,x:x+w] * my_image_color_temp[y:y+h,x:x+w]) < epsilon * np.sum(my_image_color_temp[y:y+h,x:x+w]):
           contours_new.append(contour)
           mask[y:y+h,x:x+w] = 1

    del(my_image_color_temp, mask)

    return(contours_new)
#end of cv2_get_unique_contours

#Source: https://docs.opencv.org/4.3.0/dd/d3b/tutorial_py_svm_opencv.html
#raw or spatial moments (the basis) is equal to the mean of the image
#central moments (depending on the raw moments) is equal to the variance of the image
#central standardized or normalized or scale invariant moments (based on the central moments) is equal to the skewness of the image
def cv2_deskew(img, SZ=20):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img,M,(SZ, SZ),flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR)
    return img
