import cv2
import os
import numpy as np

def my_PreProc(image):
    assert(len(image.shape) == 3)  # Ensure image is 3D
    # Convert to Grayscale
    gray_image = rgb2gray(image)
    # Apply CLAHE
    clahe_image = clahe_equalized(gray_image)
    # Adjust Gamma
    adjusted_image = adjust_gamma(clahe_image, 1.2)
    return adjusted_image

def rgb2gray(img):
    assert (len(img.shape) == 3)  # 3D array
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def clahe_equalized(img):
    assert (len(img.shape) == 2)  # 2D array for grayscale image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)

def adjust_gamma(img, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def process_images_in_directory(images_dirs, labels_dirs, output_directory):
    for dataset_name, image_dir, label_dir in zip(['chase_db1', 'drive', 'stare'], images_dirs, labels_dirs):
        # Create output directories for images and labels
        dataset_output_dir = os.path.join(output_directory, dataset_name)
        image_output_dir = os.path.join(dataset_output_dir, 'images')
        label_output_dir = os.path.join(dataset_output_dir, 'labels')
        os.makedirs(image_output_dir, exist_ok=True)
        os.makedirs(label_output_dir, exist_ok=True)

        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.endswith('.png'):
                    image_path = os.path.join(root, file)
                    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                    if image is None:
                        print(f"Warning: Could not read image at {image_path}.")
                        continue

                    processed_image = my_PreProc(image)  # Directly process the 3D image
                    
                    # Save processed image
                    output_image_path = os.path.join(image_output_dir, file)
                    cv2.imwrite(output_image_path, processed_image)
                    print(f"Processed and saved {output_image_path}.")

                    label_path = os.path.join(label_dir, file)  # Assuming labels have the same filename
                    if os.path.exists(label_path):
                        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale if needed
                        if label is not None:
                            output_label_path = os.path.join(label_output_dir, file)
                            cv2.imwrite(output_label_path, label)  # Save label image
                            print(f"Saved label image {output_label_path}.")
                        else:
                            print(f"Warning: Could not read label at {label_path}.")
                    else:
                        print(f"Warning: Label not found for {file}.")

if __name__ == '__main__':
    images_dirs = [
        '/home/s12gb1/aima/RetinalVesselSemiSeg/SkinSeg/proceeded_png/chase_db1/images',
        '/home/s12gb1/aima/RetinalVesselSemiSeg/SkinSeg/proceeded_png/drive/images',
        '/home/s12gb1/aima/RetinalVesselSemiSeg/SkinSeg/proceeded_png/stare/images'
    ]
    labels_dirs = [
        '/home/s12gb1/aima/RetinalVesselSemiSeg/SkinSeg/proceeded_png/chase_db1/labels',
        '/home/s12gb1/aima/RetinalVesselSemiSeg/SkinSeg/proceeded_png/drive/labels',
        '/home/s12gb1/aima/RetinalVesselSemiSeg/SkinSeg/proceeded_png/stare/labels'
    ]
    output_dir = '/home/s12gb1/aima/RetinalVesselSemiSeg/SkinSeg/preprocessing'
    process_images_in_directory(images_dirs, labels_dirs, output_dir)
# import cv2
# import os
# import numpy as np
# import albumentations as A

# def my_PreProc(image):
#     assert(len(image.shape) == 3)  # Ensure image is 3D
#     # Augment the image
#     augmented = get_augmentation()(image=image)
#     return augmented['image']

# def get_augmentation():
#     return A.Compose([
#         A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=0.5),
#         A.RandomGamma(gamma_limit=(80, 120), p=0.5),
#         A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.9),
#         A.Blur(blur_limit=3, p=0.5),
#         A.Normalize(mean=(0.5,), std=(0.5,), p=1),  # Normalize to 0-1 range
#         A.Lambda(image=to_tensor)
#     ])

# def to_tensor(x, **kwargs):
#     return x.astype('float32')

# def process_images_in_directory(images_dirs, labels_dirs, output_directory):
#     for dataset_name, image_dir, label_dir in zip(['chase_db1', 'drive', 'stare'], images_dirs, labels_dirs):
#         # Create output directories for images and labels
#         dataset_output_dir = os.path.join(output_directory, dataset_name)
#         image_output_dir = os.path.join(dataset_output_dir, 'images')
#         label_output_dir = os.path.join(dataset_output_dir, 'labels')
#         os.makedirs(image_output_dir, exist_ok=True)
#         os.makedirs(label_output_dir, exist_ok=True)

#         for root, dirs, files in os.walk(image_dir):
#             for file in files:
#                 if file.endswith('.png'):
#                     image_path = os.path.join(root, file)
#                     image = cv2.imread(image_path, cv2.IMREAD_COLOR)
#                     if image is None:
#                         print(f"Warning: Could not read image at {image_path}.")
#                         continue

#                     processed_image = my_PreProc(image)  # Process the 3D image
                    
#                     # Save processed image
#                     output_image_path = os.path.join(image_output_dir, file)
#                     cv2.imwrite(output_image_path, processed_image)
#                     print(f"Processed and saved {output_image_path}.")

#                     # Process corresponding label image
#                     label_path = os.path.join(label_dir, file)  # Assuming labels have the same filename
#                     if os.path.exists(label_path):
#                         label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
#                         if label is not None:
#                             output_label_path = os.path.join(label_output_dir, file)
#                             cv2.imwrite(output_label_path, label)  # Save label image
#                             print(f"Saved label image {output_label_path}.")
#                         else:
#                             print(f"Warning: Could not read label at {label_path}.")
#                     else:
#                         print(f"Warning: Label not found for {file}.")

# if __name__ == '__main__':
#     images_dirs = [
#         '/home/s12gb1/aima/RetinalVesselSemiSeg/SkinSeg/proceeded_png/chase_db1/images',
#         '/home/s12gb1/aima/RetinalVesselSemiSeg/SkinSeg/proceeded_png/drive/images',
#         '/home/s12gb1/aima/RetinalVesselSemiSeg/SkinSeg/proceeded_png/stare/images'
#     ]
#     labels_dirs = [
#         '/home/s12gb1/aima/RetinalVesselSemiSeg/SkinSeg/proceeded_png/chase_db1/labels',
#         '/home/s12gb1/aima/RetinalVesselSemiSeg/SkinSeg/proceeded_png/drive/labels',
#         '/home/s12gb1/aima/RetinalVesselSemiSeg/SkinSeg/proceeded_png/stare/labels'
#     ]
#     output_dir = '/home/s12gb1/aima/RetinalVesselSemiSeg/SkinSeg/preprocessing'
#     process_images_in_directory(images_dirs, labels_dirs, output_dir)
