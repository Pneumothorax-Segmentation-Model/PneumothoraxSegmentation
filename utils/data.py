import pandas as pd
import numpy as np
import pydicom
import os
import sys
import glob2
import shutil
from utils.mask import maskFromRLE
from PIL import Image
import cv2 as cv2
from tqdm import tqdm
from matplotlib import pyplot as plt
import copy


def generate_data(anotations_file, root, dcm_dir, img_output_dir, mask_output_dir):
    """Generate png images from dicom files and creates mask from anotations_file saving to output_dir"""

    imageIdxs = anotations_file["ImageId"]
    encodedPixels = anotations_file[" EncodedPixels"]

    # create a temp directory to store merge all dcm files in one folder
    temp_dcm_path = os.path.join(root, "temp_dcm")
    if not os.path.exists(temp_dcm_path):
        os.makedirs(temp_dcm_path)

    # copy and transfer all the files into the temporary directory
    for filename in tqdm(glob2.glob(f"{dcm_dir}/**/*.dcm")):
        fname = str(filename).split("/")[-1]
        shutil.copy(str(filename), os.path.join(temp_dcm_path, fname))

    # generate merged masks
    mask_output_path = mask_output_dir
    img_output_path = img_output_dir
    if not os.path.isdir(img_output_path):
        os.mkdir(img_output_path)
    if not os.path.isdir(mask_output_path):
        os.mkdir(mask_output_path)

    check_data = None
    temp_mask = np.zeros((1024, 1024))
    for i in tqdm(range(len(imageIdxs))):
        if encodedPixels[i].strip() != "-1":
            # image has associated mask

            ds = pydicom.read_file(os.path.join(temp_dcm_path, imageIdxs[i] + ".dcm"))
            img = ds.pixel_array
            img_mem = Image.fromarray(img)

            rleToMask = maskFromRLE(
                rle=encodedPixels[i], width=img.shape[0], height=img.shape[1]
            )
            if check_data == imageIdxs[i]:
                temp_mask += rleToMask
                mask = rotateMask(rleToMask)
                cv2.imwrite(
                    mask_output_path + f"/{imageIdxs[i]}_mask.png",
                    mask.astype("int32"),
                )
                img_mem.save(img_output_path + f"/{imageIdxs[i]}.png")
            else:
                temp_mask = rleToMask
                mask = rotateMask(rleToMask)
                cv2.imwrite(
                    mask_output_path + f"/{imageIdxs[i]}_mask.png",
                    mask.astype("int32"),
                )
                img_mem.save(img_output_path + f"/{imageIdxs[i]}.png")
        else:
            ds = pydicom.read_file(os.path.join(temp_dcm_path, imageIdxs[i] + ".dcm"))
            img = ds.pixel_array
            img_mem = Image.fromarray(img)
            img_mem.save(img_output_path + f"/{imageIdxs[i]}.png")

        check_data = imageIdxs[i]

    # remove the temporary directory
    shutil.rmtree(temp_dcm_path)
    print("mask-images-train", len(os.listdir(mask_output_path)))


def rotateMask(mask):
    """Rotates a given mask as numpy matrix to the correct orientation"""
    mask = np.rot90(mask, k=-1)
    mask = np.fliplr(mask)
    return mask


def showOverlay(img_dir, mask_dir, imageId):
    """Show overlay of mask and image using matplotib"""
    mask = cv2.imread(os.path.join(img_dir, imageId + "_mask.png"))
    img = cv2.imread(os.path.join(mask_dir, imageId + ".png"))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    alpha_mask = copy.deepcopy(mask)
    alpha_mask = alpha_mask.astype(float)
    alpha_mask[mask == 255] = 0.5
    plt.imshow(img, cmap="gray")
    plt.imshow(mask.astype(bool), alpha=alpha_mask, cmap="Blues")
    plt.show()
