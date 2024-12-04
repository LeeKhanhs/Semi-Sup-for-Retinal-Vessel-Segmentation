# data_augmentation.py

import os
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFile

class DataAugmentation:
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    @staticmethod
    def openImage(image_path):
        return Image.open(image_path)

    @staticmethod
    def randomRotation(image, label):
        random_angle = np.random.randint(1, 360)
        return image.rotate(random_angle, Image.BICUBIC), label.rotate(random_angle, Image.NEAREST)

    @staticmethod
    def randomCrop(image, label):
        image_width, image_height = image.size
        crop_win_size = np.random.randint(40, 68)
        random_region = (
            (image_width - crop_win_size) // 2,
            (image_height - crop_win_size) // 2,
            (image_width + crop_win_size) // 2,
            (image_height + crop_win_size) // 2
        )
        return image.crop(random_region), label.crop(random_region)

    @staticmethod
    def randomColor(image, label):
        random_factor = np.random.uniform(0, 3.0)
        color_image = ImageEnhance.Color(image).enhance(random_factor)
        brightness_factor = np.random.uniform(0.5, 1.5)
        brightness_image = ImageEnhance.Brightness(color_image).enhance(brightness_factor)
        contrast_factor = np.random.uniform(0.5, 1.5)
        contrast_image = ImageEnhance.Contrast(brightness_image).enhance(contrast_factor)
        sharpness_factor = np.random.uniform(0, 3.0)
        return ImageEnhance.Sharpness(contrast_image).enhance(sharpness_factor), label

    @staticmethod
    def randomGaussian(image, label, mean=0.2, sigma=0.3):
        def gaussianNoisy(im):
            return im + random.gauss(mean, sigma)

        img = np.asarray(image).copy()
        width, height = img.shape[:2]
        img[:, :, 0] = gaussianNoisy(img[:, :, 0])
        img[:, :, 1] = gaussianNoisy(img[:, :, 1])
        img[:, :, 2] = gaussianNoisy(img[:, :, 2])
        return Image.fromarray(np.uint8(img)), label

    @staticmethod
    def saveImage(image, path):
        image.save(path)
