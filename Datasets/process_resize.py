# '''
# Input: downloaded datasets
# Process: resize, change jpg to npy, store images and labels to Image/, Label/
# From https://github.com/jcwang123/BA-Transformer
# '''

# import cv2
# import os
# import numpy as np
# from tqdm import tqdm
# import matplotlib.pyplot as plt


# def process_isic2018(
#         origin_folder = './data/train',
#         dim=(512, 512), proceeded_folder='./proceeded_data'): # '/raid/wjc/data/skin_lesion/isic2018/')
#     image_dir_path = origin_folder+'/images/'    
#     mask_dir_path =  origin_folder+'/labels/'  
#     # '/raid/wl/2018_raw_data/ISIC2018_Task1_Training_GroundTruth/'

#     image_path_list = os.listdir(image_dir_path)
#     mask_path_list = os.listdir(mask_dir_path)

#     image_path_list = list(filter(lambda x: x[-3:] == 'jpg', image_path_list))
#     mask_path_list = list(filter(lambda x: x[-3:] == 'png', mask_path_list))
    
#     # align masks and inputs
#     image_path_list.sort()
#     mask_path_list.sort()

#     print('number of images: {}, number of masks: {}'.format(len(image_path_list), len(mask_path_list)))

#     # ISBI Dataset
#     for image_path, mask_path in zip(image_path_list, mask_path_list):
#         if image_path[-3:] == 'jpg':
#             print(image_path)
#             assert os.path.basename(image_path)[:-4].split(
#                 '_')[1] == os.path.basename(mask_path)[:-4].split('_')[1]
#             _id = os.path.basename(image_path)[:-4].split('_')[1]
#             image_path = os.path.join(image_dir_path, image_path)
#             mask_path = os.path.join(mask_dir_path, mask_path)
#             image = plt.imread(image_path)
#             mask = plt.imread(mask_path)
#             if len(mask.shape) == 3:
#                 mask = np.int64(np.all(mask[:, :, :3] == 1, axis=2))

#             image_new = cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)
#             mask_new = cv2.resize(mask, dim, interpolation=cv2.INTER_NEAREST)

#             save_dir_path = proceeded_folder + '/images'
#             os.makedirs(save_dir_path, exist_ok=True)
#             np.save(os.path.join(save_dir_path, _id + '.npy'), image_new)

#             save_dir_path = proceeded_folder + '/labels'
#             os.makedirs(save_dir_path, exist_ok=True)
#             np.save(os.path.join(save_dir_path, _id + '.npy'), mask_new)
            


# def process_PH2(
#     PH2_origin_folder = '/bigdata/siyiplace/data/skin_lesion/PH2_rawdata',
#     PH2_proceeded_folder = '/bigdata/siyiplace/data/skin_lesion/PH2'):
    
#     PH2_images_path = os.path.join(PH2_origin_folder,'/PH2Dataset/PH2_Dataset_images')
#     path_list = os.listdir(PH2_images_path)
#     path_list.sort()

#     for path in path_list:
#         image_path = os.path.join(PH2_images_path, path,
#                                   path + '_Dermoscopic_Image', path + '.bmp')
#         label_path = os.path.join(PH2_images_path, path, path + '_lesion',
#                                   path + '_lesion.bmp')
#         image = plt.imread(image_path)
#         label = plt.imread(label_path)
#         label = label[:, :, 0]

#         dim = (512, 512)
#         image_new = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
#         label_new = cv2.resize(label, dim, interpolation=cv2.INTER_AREA)

#         image_save_path = os.path.join(
#             PH2_proceeded_folder,'/Image',
#             path + '.npy') #  '/data2/cf_data/skinlesion_segment/PH2_rawdata/PH2/Image'
#         label_save_path = os.path.join(
#             PH2_proceeded_folder,'/Label',
#             path + '.npy') # /data2/cf_data/skinlesion_segment/PH2_rawdata/PH2/Label

#         np.save(image_save_path, image_new)
#         np.save(label_save_path, label_new)


# def process_SKD(
#     SKD_images_folder = '/bigdata/siyiplace/data/skin_lesion/skin_cancer_detection',
#     SKD_proceeded_folder = '/bigdata/siyiplace/data/skin_lesion/SKD'):
#     '''
#     SKin Cancer Detection dataset
#     '''
    
#     SKD_images_path1 = '{}/skin_image_data_set-1/Skin Image Data Set-1/skin_data/melanoma/'.format(SKD_images_folder)
#     SKD_images_path2 = '{}/skin_image_data_set-2/Skin Image Data Set-2/skin_data/notmelanoma/'.format(SKD_images_folder)

    
#     for images_path in [SKD_images_path1, SKD_images_path2]:
#         for dataset_name in ['dermis', 'dermquest']:
#             path_list = os.listdir('{}{}'.format(images_path, dataset_name))

#             for path in path_list:
#                 if path[-4:] == '.jpg':
#                     image_path = os.path.join('{}{}'.format(images_path, dataset_name), path)
#                     label_path = os.path.join('{}{}'.format(images_path, dataset_name), path[:-8]+'contour.png')
#                 else: continue

#                 image = plt.imread(image_path)
#                 label = plt.imread(label_path)
#                 dim = (512, 512)
#                 image_new = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
#                 label_new = cv2.resize(label, dim, interpolation=cv2.INTER_AREA)

#                 image_save_path = os.path.join(
#                     SKD_proceeded_folder,'/Image',
#                     dataset_name+'_'+path[:-4] + '.npy')
#                 label_save_path = os.path.join(
#                     SKD_proceeded_folder,'/Label',
#                     dataset_name+'_'+path[:-4] + '.npy') 
#                 np.save(image_save_path, image_new)
#                 np.save(label_save_path, label_new)
                

# def process_DMF(
#     DMF_images_folder = '/bigdata/siyiplace/data/skin_lesion/DMF_origin',
#     DMF_proceeded_folder = '/bigdata/siyiplace/data/skin_lesion/DMF'):
#     '''
#     Dermofit (DMF) dataset
#     '''
    
#     DMF_images_path = '{}/images'.format(DMF_images_folder)

#     path_list = os.listdir(DMF_images_path)
#     path_list.sort()

#     for path in tqdm(path_list):
#         image_path = os.path.join(DMF_images_path, path,
#                                   path + '.png')
#         label_path = os.path.join(DMF_images_path, path, path + 'mask.png')
#         image = plt.imread(image_path)
#         label = plt.imread(label_path)

#         dim = (512, 512)
#         image_new = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
#         image_new = np.clip(image_new*255, 0, 255).astype(np.uint8) if image_new.max() < 1.2 else image_new
#         label_new = cv2.resize(label, dim, interpolation=cv2.INTER_AREA)

#         image_save_path = os.path.join(
#             DMF_proceeded_folder,'/Image',
#             path + '.npy') 
#         label_save_path = os.path.join(
#             DMF_proceeded_folder,'/Label',
#             path + '.npy') 

#         np.save(image_save_path, image_new)
#         np.save(label_save_path, label_new)


# if __name__ == '__main__':
#     process_isic2018(origin_folder = './data/train', proceeded_folder='./proceeded_data/train')
#     process_isic2018(origin_folder = './data/val', proceeded_folder='./proceeded_data/val')
#     process_isic2018(origin_folder = './data/test', proceeded_folder='./proceeded_data/test')
import cv2
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import gzip
import shutil

def process_drive(origin_folder='./data/drive/training', proceeded_folder='./proceeded_data/drive'):
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

        image = plt.imread(image_path_full)
        mask = plt.imread(mask_path_full)

        # Resize to (512, 512)
        image_new = cv2.resize(image, (512, 512), interpolation=cv2.INTER_CUBIC)
        mask_new = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

        # Remove '_training' from image name and '_manual1' from mask name
        image_name = image_path.replace('_training.tif', '') + '.npy'
        mask_name = mask_path.replace('_manual1.gif', '') + '.npy'

        np.save(os.path.join(proceeded_folder, 'images', image_name), image_new)
        np.save(os.path.join(proceeded_folder, 'labels', mask_name), mask_new)


def process_stare(origin_folder='./data/stare', proceeded_folder='./proceeded_data/stare'):
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
        image = plt.imread(image_path_full[:-3])  # Remove .gz
        os.remove(image_path_full[:-3])  # Optionally remove the uncompressed file

        # Resize to (512, 512)
        image_new = cv2.resize(image, (512, 512), interpolation=cv2.INTER_CUBIC)

        # Save the processed image
        np.save(os.path.join(proceeded_folder, 'images', image_path[:-7] + '.npy'), image_new)  # Remove .ppm.gz

        # Process the corresponding mask
        mask_path_full = os.path.join(mask_dir_path, image_path[:-7] + '.ah.ppm.gz')  # Get corresponding mask path

        # Decompress the .gz files for mask
        with gzip.open(mask_path_full, 'rb') as f_in:
            with open(mask_path_full[:-3], 'wb') as f_out:  # Remove .gz
                shutil.copyfileobj(f_in, f_out)

        mask = plt.imread(mask_path_full[:-3])  # Remove .gz
        os.remove(mask_path_full[:-3])  # Optionally remove the uncompressed file

        # Resize to (512, 512)
        mask_new = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

        # Save the processed mask
        np.save(os.path.join(proceeded_folder, 'labels', image_path[:-7] + '.npy'), mask_new)  # Remove .ppm.gz


def process_chase_db1(origin_folder='./data/chase_db1', proceeded_folder='./proceeded_data/chase_db1'):
    image_dir_path = os.path.join(origin_folder, 'images/')
    mask_dir_path = os.path.join(origin_folder, 'labels/')

    image_path_list = os.listdir(image_dir_path)
    mask_path_list_1st = list(filter(lambda x: x.endswith('_1stHO.png'), os.listdir(mask_dir_path)))

    os.makedirs(os.path.join(proceeded_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(proceeded_folder, 'labels'), exist_ok=True)

    for image_path in image_path_list:
        image_path_full = os.path.join(image_dir_path, image_path)
        image = plt.imread(image_path_full)

        # Resize to (512, 512)
        image_new = cv2.resize(image, (512, 512), interpolation=cv2.INTER_CUBIC)

        np.save(os.path.join(proceeded_folder, 'images', image_path[:-4] + '.npy'), image_new)

    for mask_path in mask_path_list_1st:
        mask_path_full = os.path.join(mask_dir_path, mask_path)
        mask = plt.imread(mask_path_full)

        # Resize to (512, 512)
        mask_new = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        
        # Remove both _1stHO and _1 from the filename
        new_mask_filename = mask_path.replace('_1stHO.png', '').replace('_1.png', '') + '.npy'
        np.save(os.path.join(proceeded_folder, 'labels', new_mask_filename), mask_new)


if __name__ == '__main__':
    # Process the three datasets
    process_drive()
    process_stare()
    process_chase_db1()
