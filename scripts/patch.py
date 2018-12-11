import os
import random
import numpy as np

from deblurgan.utils import load_image, list_image_files


def preprocess_image(images_group):
    images_group = [np.array(image) for image in images_group]
    patch = random_patch(dim_patch=[256, 256], dim_image=images_group[0].shape)
    return [(image[patch] - 127.5) / 127.5 for image in images_group]


def load_images(path, n_images):
    if n_images < 0:
        n_images = float("inf")
    A_paths, B_paths = os.path.join(path, 'A'), os.path.join(path, 'B')
    all_A_paths, all_B_paths = list_image_files(A_paths), list_image_files(B_paths)
    p = np.random.permutation(len(all_A_paths))
    all_A_paths, all_B_paths = np.array(all_A_paths)[p], np.array(all_B_paths)[p]
    images_A, images_B = [], []
    images_A_paths, images_B_paths = [], []
    for path_A, path_B in zip(all_A_paths, all_B_paths):
        img_A, img_B = load_image(path_A), load_image(path_B)
        image_A, image_B = preprocess_image([img_A, img_B])
        images_A.append(image_A)
        images_B.append(image_B)
        images_A_paths.append(path_A)
        images_B_paths.append(path_B)
        if len(images_A) > n_images - 1: break
    return {
        'A': np.array(images_A),
        'A_paths': np.array(images_A_paths),
        'B': np.array(images_B),
        'B_paths': np.array(images_B_paths)
    }


def random_patch(dim_patch=[256, 256], dim_image=[720, 1280]):
    slices = []
    for dim_patch_i, dim_image_i in zip(dim_patch, dim_image):
        dim_image_i_start = np.random.randint(dim_image_i - dim_patch_i)
        dim_image_i_end = dim_image_i_start + dim_patch_i
        slices.append(np.s_[dim_image_i_start:dim_image_i_end])
    return tuple([*slices, np.s_[:]])
