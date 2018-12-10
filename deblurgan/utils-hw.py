import os
from PIL import Image
import numpy as np
import tensorflow as tf

LOW = (64,64)
RESHAPE = (256,256)

def is_an_image_file(filename):
    IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg']
    for ext in IMAGE_EXTENSIONS:
        if ext in filename:
            return True
    return False


def list_image_files(directory):
    files = sorted(os.listdir(directory))
    return [os.path.join(directory, f) for f in files if is_an_image_file(f)]


def load_image(path):
    img = Image.open(path)
    return img

def preprocess_image(cv_img):
    cv_img = cv_img.resize(RESHAPE)
    img = np.array(cv_img)
    img = (img - 127.5) / 127.5
    return img

def preprocess_low_res_images(cv_img):
    cv_img = cv_img.resize(LOW)
    cv_img = cv_img.resize(RESHAPE)
    img = np.array(cv_img)
    img = (img - 127.5) / 127.5
    return img

def deprocess_image(img):
    img = img * 127.5 + 127.5
    return img.astype('uint8')


def save_image(np_arr, path):
    img = np_arr * 127.5 + 127.5
    im = Image.fromarray(img)
    im.save(path)

def load_images(path, n_images):
    if n_images < 0:
        n_images = float("inf")
    all_paths = list_image_files(path)
    images_A, images_B = [],[]
    
    for path in all_paths:
        img = load_image(path)
        images_A.append(preprocess_low_res_images(img))
        images_B.append(preprocess_image(img))
        
        if len(images_A) > n_images - 1: break

    return {
        'A': np.array(images_A),
       
        'B': np.array(images_B),

    }   

