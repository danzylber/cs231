import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import math
import sklearn
from sklearn import mixture
from sklearn.utils import shuffle
from skimage import measure
from glob import glob
import os
from imgaug import augmenters as iaa
from imgaug import parameters as iap
import imgaug as ia

from subprocess import check_output

# Get images and ids
TRAIN_DATA = "train"
type_1_files = glob(os.path.join(TRAIN_DATA, "Type_1", "*.jpg"))
type_1_ids = np.array([s[len(os.path.join(TRAIN_DATA, "Type_1"))+1:-4] for s in type_1_files])
type_2_files = glob(os.path.join(TRAIN_DATA, "Type_2", "*.jpg"))
type_2_ids = np.array([s[len(os.path.join(TRAIN_DATA, "Type_2"))+1:-4] for s in type_2_files])
type_3_files = glob(os.path.join(TRAIN_DATA, "Type_3", "*.jpg"))
type_3_ids = np.array([s[len(os.path.join(TRAIN_DATA, "Type_3"))+1:-4] for s in type_3_files])

# Function to get the file name given ID and Type
def get_filename(image_id, image_type):
    """
    Method to get image file path from its id and type
    """
    if image_type == "Type_1" or \
        image_type == "Type_2" or \
        image_type == "Type_3":
        data_path = os.path.join(TRAIN_DATA, image_type)
    elif image_type == "Test":
        data_path = TEST_DATA
    elif image_type == "AType_1" or \
          image_type == "AType_2" or \
          image_type == "AType_3":
        data_path = os.path.join(ADDITIONAL_DATA, image_type)
    else:
        raise Exception("Image type '%s' is not recognized" % image_type)

    ext = 'jpg'
    return os.path.join(data_path, "{}.{}".format(image_id, ext))

# Function to get the image given ID and Type
def get_image_data(image_id, image_type):
    """
    Method to get image data as np.array specifying image id and type
    """
    fname = get_filename(image_id, image_type)
    img = cv2.imread(fname)
    assert img is not None, "Failed to read image : %s, %s" % (image_id, image_type)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# Functions to perform the cropping, based on the algorithm described on the links:
# https://www.youtube.com/watch?v=g8bSdXCG-lA
# https://www.youtube.com/watch?v=VNbkzsnllsU
def maxHist(hist):
    maxArea = (0, 0, 0)
    height = []
    position = []
    for i in range(len(hist)):
        if (len(height) == 0):
            if (hist[i] > 0):
                height.append(hist[i])
                position.append(i)
        else:
            if (hist[i] > height[-1]):
                height.append(hist[i])
                position.append(i)
            elif (hist[i] < height[-1]):
                while (height[-1] > hist[i]):
                    maxHeight = height.pop()
                    area = maxHeight * (i - position[-1])
                    if (area > maxArea[0]):
                        maxArea = (area, position[-1], i)
                    last_position = position.pop()
                    if (len(height) == 0):
                        break
                position.append(last_position)
                if (len(height) == 0):
                    height.append(hist[i])
                elif (height[-1] < hist[i]):
                    height.append(hist[i])
                else:
                    position.pop()
    while (len(height) > 0):
        maxHeight = height.pop()
        last_position = position.pop()
        area = maxHeight * (len(hist) - last_position)
        if (area > maxArea[0]):
            maxArea = (area, len(hist), last_position)
    return maxArea

def maxRect(img):
    maxArea = (0, 0, 0)
    addMat = np.zeros(img.shape)
    for r in range(img.shape[0]):
        if r == 0:
            addMat[r] = img[r]
            area = maxHist(addMat[r])
            if area[0] > maxArea[0]:
                maxArea = area + (r,)
        else:
            addMat[r] = img[r] + addMat[r - 1]
            addMat[r][img[r] == 0] *= 0
            area = maxHist(addMat[r])
            if area[0] > maxArea[0]:
                maxArea = area + (r,)
    return (
    int(maxArea[3] + 1 - maxArea[0] / abs(maxArea[1] - maxArea[2])), maxArea[2], maxArea[3], maxArea[1], maxArea[0])

def cropCircle(img):
    if (img.shape[0] > img.shape[1]):
        tile_size = (int(img.shape[1] * 256 / img.shape[0]), 256)
    else:
        tile_size = (256, int(img.shape[0] * 256 / img.shape[1]))

    img = cv2.resize(img, dsize=tile_size)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY);
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    _, contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    main_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    ff = np.zeros((gray.shape[0], gray.shape[1]), 'uint8')
    cv2.drawContours(ff, main_contour, -1, 1, 15)
    ff_mask = np.zeros((gray.shape[0] + 2, gray.shape[1] + 2), 'uint8')
    cv2.floodFill(ff, ff_mask, (int(gray.shape[1] / 2), int(gray.shape[0] / 2)), 1)
    # cv2.circle(ff, (int(gray.shape[1]/2), int(gray.shape[0]/2)), 3, 3, -1)

    rect = maxRect(ff)
    img_crop = img[min(rect[0], rect[2]):max(rect[0], rect[2]), min(rect[1], rect[3]):max(rect[1], rect[3])]
    cv2.rectangle(ff, (min(rect[1], rect[3]), min(rect[0], rect[2])), (max(rect[1], rect[3]), max(rect[0], rect[2])), 3,
                  2)

    # plt.subplot(121)
    # plt.imshow(img)
    # plt.subplot(122)
    # plt.imshow(ff)
    # plt.show()

    return img_crop


# Segmentation of portions in the middle of the image and that are also very red
# Function to build the features
def Ra_space(img, Ra_ratio, a_threshold_1, a_threshold_2):
    imgLab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB);
    w = img.shape[0]
    h = img.shape[1]
    Ra = np.zeros((w * h, 3))
    for i in range(w):
        for j in range(h):
            R = math.sqrt((w / 2 - i) * (w / 2 - i) + (h / 2 - j) * (h / 2 - j))
            Ra[i * h + j, 0] = R
            Ra[i * h + j, 1] = min(imgLab[i][j][1], a_threshold_1)
            Ra[i * h + j, 2] = min(imgLab[i][j][1], a_threshold_2)

    Ra[:, 0] /= max(Ra[:, 0])
    Ra[:, 0] *= Ra_ratio
    Ra[:, 1] /= max(Ra[:, 1])
    Ra[:, 2] /= max(Ra[:, 2])

    return Ra

# Function to crop the image
def cropCentralRed(img):
    toPlotImg = img
    img = cropCircle(img)
    w = img.shape[0]
    h = img.shape[1]

    # Saturate the a-channel at 150
    Ra = Ra_space(img, 1.0, 150, 170)
    a_channel_1 = np.reshape(Ra[:, 1], (w, h))
    a_channel_2 = np.reshape(Ra[:, 2], (w, h))

    # Run the clustering
    g = mixture.GaussianMixture(n_components=2, covariance_type='diag', random_state=0, init_params='kmeans')
    image_array_sample = shuffle(Ra, random_state=0)[:1000]
    g.fit(image_array_sample)
    labels = g.predict(Ra)
    labels += 1  # Add 1 to avoid labeling as 0 since regionprops ignores the 0-label.

    # The cluster that has the highest a-mean is selected.
    labels_2D = np.reshape(labels, (w, h))
    gg_labels_regions = measure.regionprops(labels_2D, intensity_image=a_channel_2)
    gg_intensity = [prop.mean_intensity for prop in gg_labels_regions]
    cervix_cluster = gg_intensity.index(max(gg_intensity)) + 1

    # Get a mask for the cluster
    mask = np.zeros((w * h, 1), 'uint8')
    mask[labels == cervix_cluster] = 255
    mask_2D = np.reshape(mask, (w, h))

    rect = cv2.boundingRect(mask_2D)
    cropped_img = img[rect[0]:(rect[0] + rect[2]), rect[1]:(rect[1] + rect[3])]

    return cropped_img

    # plt.subplot(221)
    # plt.imshow(toPlotImg)
    # plt.subplot(222)
    # plt.imshow(a_channel_1)
    # plt.subplot(223)
    # plt.imshow(a_channel_2)
    # plt.subplot(224)
    # plt.imshow(cropped_img)
    # plt.show()

#cropCentralRed(get_image_data(0, "Type_1"))

# Data augmentation
def augment_one_image_deterministic(img):
    augmentedImages = []

    # Build the sequence of augmentations (all possible combinations)
    for gaussianBlur in [0, 20.0]:
        for (scale_X, scale_Y) in [(0.8, 0.8), (1, 1), (1.25, 1.25)]:
            for fliplr in [True, False]:
                for shearVal in [-16, 0, 16]:
                    for multiply in [0.5, 1, 1.5]:
                        # for (translate_X, translate_Y) in [(-32, 0), (0, 0), (32,0), (0, 32), (0, -32)]:
                        # for rotationAngle in [-45, 0, 45]:

                        if shearVal < 0 or multiply < 1 or scale_X < 1:
                            continue

                        # Start with an empty list of augmentations
                        seq = []

                        # Gaussian noise
                        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.2), per_channel=0.5)

                        # Gaussian Blur
                        seq.append(iaa.GaussianBlur(gaussianBlur))

                        # Scale
                        seq.append(iaa.Affine(scale={"x": scale_X, "y": scale_Y}))

                        # Flip horizontally
                        if fliplr:
                            seq.append(iaa.Fliplr(1))

                        # Shear
                        seq.append(iaa.Affine(shear=shearVal))

                        # Color
                        seq.append(iaa.Multiply(multiply))

                        # Translate
                        # seq.append(iaa.Affine(translate_px={"x": (translate_X), "y": (translate_Y)}))

                        # Rotate
                        # seq.append(iaa.Affine(rotate = rotationAngle))

                        # Create the augmentations object
                        seq = iaa.Sequential(seq)
                        #seq.show_grid(img, cols=4, rows=4)
                        augmentedImages.append(seq.augment_image(img))

    return augmentedImages

def augment_one_image_stochastic(img, num_imgs):
    augmentedImages = [img]

    for i in range(num_imgs):
        seq = iaa.Sequential([
            iaa.Fliplr(iap.Uniform(0, 1)),
            iaa.GaussianBlur(iap.Uniform(0, 20.0)),
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.2)),
            iaa.Add(iap.Uniform(-10, 10)),  # change brightness of images (by -10 to 10 of original value)
            iaa.Multiply(iap.Uniform(0.5, 1.5)),  # change brightness of images (50-150% of original value)
            iaa.ContrastNormalization(iap.Uniform(0.5, 2)),  # improve or worsen the contrast
            iaa.ContrastNormalization(iap.Uniform(0.5, 2)),  # improve or worsen the contrast
            iaa.Affine(
                scale={"x": iap.Uniform(0.8, 1.2), "y": iap.Uniform(0.8, 1.2)},# scale images to 80-120% of their size, individually per axis
                translate_px={"x": iap.Uniform(-16, 16), "y": iap.Uniform(-16, 16)},  # translate by -16 to +16 pixels (per axis)
                rotate=iap.Uniform(-10, 10),  # rotate by -45 to +45 degrees
                shear=iap.Uniform(-16, 16)#,  # shear by -16 to +16 degrees
            ),
            iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)# apply elastic transformations with random strengths
        ])

        augmentedImages.append(seq.augment_image(img))

    return augmentedImages

image = get_image_data(7, 'Type_1')

augmentedImages = augment_one_image_stochastic(image, 8)

i = 0
for img in augmentedImages:
    plt.subplot(3, 3, 1+i)
    plt.imshow(img)
    i+=1
plt.show()



