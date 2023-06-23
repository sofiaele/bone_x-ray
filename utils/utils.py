import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import imagesize
import numpy as np
from load_dataset import split_data

def caclulate_mean_std(dataframe):
    files = dataframe['path'].values.tolist()
    mean = np.array([0., 0., 0.])
    stdTemp = np.array([0., 0., 0.])
    std = np.array([0., 0., 0.])

    numSamples = len(files)

    for i in range(numSamples):
        im = cv2.imread(str(files[i]))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = im.astype(float) / 255.

        for j in range(3):
            mean[j] += np.mean(im[:, :, j])

    mean = (mean / numSamples)

    for i in range(numSamples):
        im = cv2.imread(str(files[i]))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = im.astype(float) / 255.
        for j in range(3):
            stdTemp[j] += ((im[:, :, j] - mean[j]) ** 2).sum() / (im.shape[0] * im.shape[1])

    std = np.sqrt(stdTemp / numSamples)
    return mean, std


data_path = "new_dataset.csv"
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

optimal_image_size()