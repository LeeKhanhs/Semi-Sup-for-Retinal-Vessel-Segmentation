import cv2
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import gzip
import shutil

def process_drive(origin_folder='./data/drive/training', proceeded_folder='./proceeded_png/drive'):
    image_dir_path = os.path.join(origin_folder, 'images/')
    mask_dir_path = os.path.join(origin_folder, 'labels/')

    image_path_list = os.listdir(image_dir_path)
    mask_path_list = os.listdir(mask_dir_path)

    # Filter for .tif and .gif files
    image_path_list = list(filter(lambda x: x.endswith('_training.tif'), image_path_list))
    mask_path_list = list(filter(lambda x: x.endswith('_manual1.gif'), mask_path_list))

    # Sort files
    image_path_list.sort()
    mask_path_list.sort()

    os.makedirs(os.path.join(proceeded_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(proceeded_folder, 'labels'), exist_ok=True)

    for image_path, mask_path in zip(image_path_list, mask_path_list):
        image_path_full = os.path.join(image_dir_path, image_path)
        mask_path_full = os.path.join(mask_dir_path, mask_path)

        # Read images
        image = cv2.imread(image_path_full, cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(mask_path_full, cv2.IMREAD_UNCHANGED)

        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)  # Chuyển đổi nếu mask là hình ảnh màu

        # Resize to (512, 512)
        image_new = cv2.resize(image, (512, 512), interpolation=cv2.INTER_CUBIC)
        mask_new = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

        # Remove '_training' from image name and '_manual1' from mask name
        image_name = image_path.replace('_training.tif', '') + '.png'
        mask_name = mask_path.replace('_manual1.gif', '') + '.png'

        cv2.imwrite(os.path.join(proceeded_folder, 'images', image_name), image_new)
        cv2.imwrite(os.path.join(proceeded_folder, 'labels', mask_name), mask_new)

def process_stare(origin_folder='./data/stare', proceeded_folder='./proceeded_png/stare'):
    image_dir_path = os.path.join(origin_folder, 'images/')
    mask_dir_path = os.path.join(origin_folder, 'labels/')

    image_path_list = os.listdir(image_dir_path)
    mask_path_list = os.listdir(mask_dir_path)

    # Filter for .ppm.gz files
    image_path_list = list(filter(lambda x: x.endswith('.ppm.gz'), image_path_list))
    mask_path_list = list(filter(lambda x: x.endswith('.ah.ppm.gz'), mask_path_list))

    # Sort files
    image_path_list.sort()
    mask_path_list.sort()

    os.makedirs(os.path.join(proceeded_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(proceeded_folder, 'labels'), exist_ok=True)

    for image_path in image_path_list:
        image_path_full = os.path.join(image_dir_path, image_path)

        # Decompress the .gz files
        with gzip.open(image_path_full, 'rb') as f_in:
            with open(image_path_full[:-3], 'wb') as f_out:  # Remove .gz
                shutil.copyfileobj(f_in, f_out)

        # Read images
        image = cv2.imread(image_path_full[:-3], cv2.IMREAD_UNCHANGED)  # Remove .gz
        os.remove(image_path_full[:-3])  # Remove the uncompressed file

        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to (512, 512)
        image_new = cv2.resize(image, (512, 512), interpolation=cv2.INTER_CUBIC)

        # Save the processed image
        cv2.imwrite(os.path.join(proceeded_folder, 'images', image_path[:-7] + '.png'), image_new)  # Remove .ppm.gz

        # Process the corresponding mask
        mask_path_full = os.path.join(mask_dir_path, image_path[:-7] + '.ah.ppm.gz')

        # Decompress the .gz files for mask
        with gzip.open(mask_path_full, 'rb') as f_in:
            with open(mask_path_full[:-3], 'wb') as f_out:  # Remove .gz
                shutil.copyfileobj(f_in, f_out)

        mask = cv2.imread(mask_path_full[:-3], cv2.IMREAD_UNCHANGED)  # Remove .gz
        os.remove(mask_path_full[:-3])  # Remove the uncompressed file

        # Convert to RGB
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        # Resize to (512, 512)
        mask_new = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

        # Save the processed mask
        cv2.imwrite(os.path.join(proceeded_folder, 'labels', image_path[:-7] + '.png'), mask_new)  # Remove .ppm.gz

def process_chase_db1(origin_folder='./data/chase_db1', proceeded_folder='./proceeded_png/chase_db1'):
    image_dir_path = os.path.join(origin_folder, 'images/')
    mask_dir_path = os.path.join(origin_folder, 'labels/')

    image_path_list = os.listdir(image_dir_path)
    mask_path_list_1st = list(filter(lambda x: x.endswith('_1stHO.png'), os.listdir(mask_dir_path)))

    os.makedirs(os.path.join(proceeded_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(proceeded_folder, 'labels'), exist_ok=True)

    for image_path in image_path_list:
        image_path_full = os.path.join(image_dir_path, image_path)
        image = cv2.imread(image_path_full, cv2.IMREAD_UNCHANGED)  # Read with unchanged color

        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to (512, 512)
        image_new = cv2.resize(image, (512, 512), interpolation=cv2.INTER_CUBIC)

        cv2.imwrite(os.path.join(proceeded_folder, 'images', image_path[:-4] + '.png'), image_new)

    for mask_path in mask_path_list_1st:
        mask_path_full = os.path.join(mask_dir_path, mask_path)
        mask = cv2.imread(mask_path_full, cv2.IMREAD_UNCHANGED)  # Read with unchanged color

        # Convert to RGB
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        # Resize to (512, 512)
        mask_new = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        
        # Remove both _1stHO and _1 from the filename
        new_mask_filename = mask_path.replace('_1stHO.png', '').replace('_1.png', '') + '.png'
        cv2.imwrite(os.path.join(proceeded_folder, 'labels', new_mask_filename), mask_new)

if __name__ == '__main__':
    # Process the three datasets
    process_drive()
    process_stare()
    process_chase_db1()
