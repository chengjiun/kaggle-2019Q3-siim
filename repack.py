import os
import cv2
from glob import glob
import pydicom
from tqdm import tqdm
import zipfile
import pandas as pd
import numpy as np
from skimage import exposure
from TOOLS.mask_functions import rle2mask


def convert_images(filename, arch_out, sz=256):
    ds = pydicom.read_file(str(filename))
    img = ds.pixel_array
    img = cv2.resize(img, (sz, sz))
    img = exposure.equalize_adapthist(img)  # contrast correction
    x_tot = img.mean()  # image statistics
    x2_tot = (img ** 2).mean()
    img = ((img * 255)).clip(0, 255).astype(np.uint8)
    output = cv2.imencode(".png", img)[1]
    name = filename.split("/")[-1][:-4] + ".png"
    arch_out.writestr(name, output)
    return x_tot, x2_tot


def get_stats(stats):  # get dataset statistics
    x_tot, x2_tot = 0.0, 0.0
    for x, x2 in stats:
        x_tot += x
        x2_tot += x2

    img_avr = x_tot / len(stats)
    img_std = np.sqrt(x2_tot / len(stats) - img_avr ** 2)
    print("mean:", img_avr, ", std:", img_std)


def read_rle_to_list():
    df = pd.read_csv("input/train-rle.csv").set_index("ImageId")
    idxs = set(df.index)
    train_names = []
    for f in train:  # remove images without labels
        name = f.split("/")[-1][:-4]
        if name in idxs:
            train_names.append(f)
    return train_names, idxs, df


def convert_dicom_to_png(size, train_names, test, train_out, test_out):
    print(f"convert to size: {size}")
    trn_stats = []
    with zipfile.ZipFile(train_out, "w") as arch:
        for fname in tqdm(train_names, total=len(train_names)):
            trn_stats.append(convert_images(fname, arch, sz=size))

    test_stats = []
    with zipfile.ZipFile(test_out, "w") as arch:
        for fname in tqdm(test, total=len(test)):
            test_stats.append(convert_images(fname, arch, sz=size))
    return trn_stats, test_stats


def generate_mask_file(mask_out, idxs, df):
    mask_coverage = []
    mask_count = 0
    with zipfile.ZipFile(mask_out, "w") as arch:
        for idx in tqdm(idxs):
            masks = df.loc[idx, " EncodedPixels"]
            img = np.zeros((sz0, sz0))
            # do conversion if mask is not " -1"
            if type(masks) != str or (type(masks) == str and masks != " -1"):
                if type(masks) == str:
                    masks = [masks]
                else:
                    masks = masks.tolist()
                mask_count += 1
                for mask in masks:
                    img += rle2mask(mask, sz0, sz0).T
            mask_coverage.append(img.mean())
            img = cv2.resize(img, (sz, sz))
            output = cv2.imencode(".png", img)[1]
            name = idx + ".png"
            arch.writestr(name, output)
    print("mask coverage:", np.mean(mask_coverage) / 255, ", mask count:", mask_count)


if __name__ == "__main__":

    sz = 512
    sz0 = 1024
    PATH_TRAIN = "input/dicom-images-train/"
    PATH_TEST = "input/dicom-images-test/"
    train_out = "input/train_512.zip"
    test_out = "input/test_512.zip"
    mask_out = "input/masks_512.zip"
    train = glob(os.path.join(PATH_TRAIN, "*/*/*.dcm"))
    test = glob(os.path.join(PATH_TEST, "*/*/*.dcm"))

    # read annotation file
    train_names, idxs, df = read_rle_to_list()
    # convert to png
    trn_stats, test_stats = convert_dicom_to_png(
        sz, train_names, test, train_out, test_out
    )

    print("train set statistics:", get_stats(trn_stats))
    print("test set statistics:", get_stats(test_stats))

    # write mask
    generate_mask_file(mask_out, idxs, df)
