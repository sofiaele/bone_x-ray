import cv2
import pandas as pd
import matplotlib.pyplot as plt
import imagesize
import numpy as np
import sys
import math
from tqdm import tqdm
from numpy import asarray
from PIL import Image
import cv2 as cv
import os


def alternative_crop(data_csv):
    dataframe = pd.read_csv(data_csv)
    files = dataframe['path'].values.tolist()
    for file in files:
        gray = cv.imread(file, cv.IMREAD_GRAYSCALE)
        img = gray
        # Determine a threshold value to separate the frame from the content
        threshold_value = np.mean(gray)

        # Apply thresholding to separate the frame
        _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            # Select the contour with the largest area
            largest_contour = max(contours, key=cv2.contourArea)

            # Get the bounding box coordinates of the contour
            x, y, width, height = cv2.boundingRect(largest_contour)

            # Create a mask of the black frame using the contour
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [largest_contour], 0, 255, -1)

            # Bitwise AND the original image with the mask to align the main part with the black frame
            aligned_image = cv2.bitwise_and(img, img, mask=mask)

            # Remove the black frame by setting the pixels outside the contour to white
            aligned_image[np.where(mask == 0)] = 255

            final_image = aligned_image

            # Return the original image if no contours found or unsuccessful cropping
        else:
            final_image = img
        new_path = "/".join(file.split("/")[1:])
        new_path = "alternative_crop_dataset/" + new_path
        if not os.path.exists("/".join(new_path.split("/")[:-1])):
            # if the demo_folder directory is not present
            # then create it.
            os.makedirs("/".join(new_path.split("/")[:-1]))
        # print(new_path)
        cv.imwrite(new_path, final_image)

def crop_black_frame(data_csv):
    dataframe = pd.read_csv(data_csv)
    files = dataframe['path'].values.tolist()
    for file in files:
        gray = cv.imread(file, cv.IMREAD_GRAYSCALE)
        img = gray
        # Determine a threshold value to separate the frame from the content
        threshold_value = np.mean(gray)

        # Apply thresholding to separate the frame
        _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            # Select the contour with the largest area
            largest_contour = max(contours, key=cv2.contourArea)

            # Get the bounding box coordinates of the contour
            x, y, width, height = cv2.boundingRect(largest_contour)

            # Crop the image using the bounding box coordinates
            cropped_image = img[y:y + height, x:x + width]

        else:
            # Return the original image if no contours found or unsuccessful cropping
            cropped_image = img

        new_path = "/".join(file.split("/")[1:])
        new_path = "cropped_black_frame_dataset/" + new_path
        if not os.path.exists("/".join(new_path.split("/")[:-1])):
            # if the demo_folder directory is not present
            # then create it.
            os.makedirs("/".join(new_path.split("/")[:-1]))
        # print(new_path)
        cv.imwrite(new_path, cropped_image)

def equalize_images(data_csv):
    dataframe = pd.read_csv(data_csv)
    files = dataframe['path'].values.tolist()
    for file in files:
        img = cv.imread(file, cv.IMREAD_GRAYSCALE)
        assert img is not None, "file could not be read, check with os.path.exists()"
        equ = cv.equalizeHist(img)
        #res = np.hstack((img, equ))  # stacking images side-by-side
        new_path = "/".join(file.split("/")[1:])
        new_path = "equalized_dataset/" + new_path
        if not os.path.exists("/".join(new_path.split("/")[:-1])):
            # if the demo_folder directory is not present
            # then create it.
            os.makedirs("/".join(new_path.split("/")[:-1]))
        #print(new_path)
        cv.imwrite(new_path, equ)

equalize_images("valid.csv")
def caclulate_mean_std(dataframe):
    files = dataframe['path'].values.tolist()
    '''mean = np.array([0., 0., 0.])
    stdTemp = np.array([0., 0., 0.])'''
    means = []
    stds = []

    numSamples = len(files)
    print("-> Loading train images to calculate mean...")
    for i in tqdm(range(numSamples)):
        '''im = cv2.imread(str(files[i]))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = im.astype(float) / 255.
        for j in range(3):
            mean[j] += np.mean(im[:, :, j])'''

        # load image
        image = Image.open(str(files[i]))
        pixels = np.array(image)/255.
        # convert from integers to floats
        #pixels = pixels.astype('float32')
        # calculate per-channel means and standard deviations
        means.append(np.mean(pixels))


    mean = np.mean(means)
    squared_diff_sum = 0
    num_pixels = 0

    # Calculate the squared difference from the mean
    for i in tqdm(range(numSamples)):
        image = Image.open(str(files[i]))
        pixels = (np.array(image)/255.).flatten()
        squared_diff_sum += np.sum((pixels - mean) ** 2)
        num_pixels += pixels.size

    # Calculate the variance
    variance = squared_diff_sum / num_pixels

    # Calculate the standard deviation
    std = np.sqrt(variance)
    print('Mean: %s, Std: %s' % (mean, std))


    '''mean = (mean / numSamples)

    print("-> Loading train images to calculate std...")
    for i in tqdm(range(numSamples)):
        im = cv2.imread(str(files[i]))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = im.astype(float) / 255.
        for j in range(3):
            stdTemp[j] += ((im[:, :, j] - mean[j]) ** 2).sum() / (im.shape[0] * im.shape[1])

    std = np.sqrt(stdTemp / numSamples)'''
    return mean, std


'''train = pd.read_csv("train.csv")
mean, std = caclulate_mean_std(train)'''

def optimal_image_size():
    # Identify Image Resolutions
    dataframe = pd.read_csv("train.csv")
    # Get the Image Resolutions
    imgs = dataframe['path'].values.tolist()
    img_meta = {}
    for f in imgs: img_meta[str(f)] = imagesize.get(f)

    # Convert it to Dataframe and compute aspect ratio
    img_meta_df = pd.DataFrame.from_dict([img_meta]).T.reset_index().set_axis(['FileName', 'Size'], axis='columns')
    img_meta_df[["Width", "Height"]] = pd.DataFrame(img_meta_df["Size"].tolist(), index=img_meta_df.index)
    img_meta_df["Aspect Ratio"] = round(img_meta_df["Width"] / img_meta_df["Height"], 2)

    print(f'Total Nr of Images in the dataset: {len(img_meta_df)}')
    print(img_meta_df['Width'])
    print(img_meta_df['Height'])
    print(img_meta_df['Aspect Ratio'])

    # Visualize Image Resolutions

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    points = ax.scatter(img_meta_df.Width, img_meta_df.Height, color='blue', alpha=0.5,
                        s=img_meta_df["Aspect Ratio"] * 100, picker=True)
    ax.set_title("Image Resolution")
    ax.set_xlabel("Width", size=14)
    ax.set_ylabel("Height", size=14)
    plt.show()
    plt.clf()
    x = img_meta_df['Height'].values.tolist()
    y = img_meta_df['Width'].values.tolist()
    plt.hist(x, bins=512, alpha=0.5, label='Height')
    plt.hist(y, bins=512, alpha=0.5, label='Width')
    plt.legend(loc='upper right')
    plt.show()

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_validation_f1 = 0.0

    def early_stop(self, validation_f1):
        if validation_f1 > self.max_validation_f1:
            self.max_validation_f1 = validation_f1
            self.counter = 0
        elif validation_f1 < (self.max_validation_f1 + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def progress(loss, epoch, batch, batch_size, dataset_size):
    """
    Print the progress of the training for each epoch
    """
    batches = math.ceil(float(dataset_size) / batch_size)
    count = batch * batch_size
    bar_len = 40
    filled_len = int(round(bar_len * count / float(dataset_size)))

    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    status = 'Epoch {}, Loss: {:.4f}'.format(epoch, loss)
    _progress_str = "\r \r [{}] ...{}".format(bar, status)
    sys.stdout.write(_progress_str)
    sys.stdout.flush()

    if batch == batches:
        print()

def delete_corrupted_images():
    dataframe = pd.read_csv("train.csv")
    df = dataframe[dataframe.path != 'MURA-v1.1/train/XR_SHOULDER/patient02455/study1_negative/image3.png']
    df.to_csv("train.csv")

# Softmax function
def softmax(x):
    e_x = np.exp(x - np.max(x))  # Subtracting max value for numerical stability
    return e_x / np.sum(e_x)


