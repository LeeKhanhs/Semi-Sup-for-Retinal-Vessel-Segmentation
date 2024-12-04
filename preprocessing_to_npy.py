# import numpy as np
# import os
# import cv2

# def save_images_and_labels_as_npy(images_dirs, labels_dirs, output_npy_dir):
#     os.makedirs(output_npy_dir, exist_ok=True)  # Tạo thư mục đầu ra nếu chưa tồn tại

#     for dataset_name, image_dir, label_dir in zip(['chase_db1', 'drive', 'stare'], images_dirs, labels_dirs):
#         dataset_image_dir = os.path.join(output_npy_dir, dataset_name, 'images')
#         dataset_label_dir = os.path.join(output_npy_dir, dataset_name, 'labels')
        
#         os.makedirs(dataset_image_dir, exist_ok=True)
#         os.makedirs(dataset_label_dir, exist_ok=True)

#         for root, dirs, files in os.walk(image_dir):
#             for file in files:
#                 if file.endswith('.png'):
#                     image_path = os.path.join(root, file)
#                     image = cv2.imread(image_path, cv2.IMREAD_COLOR)
#                     if image is not None:
#                         # Chuyển từ BGR sang RGB
#                         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#                         image_npy_path = os.path.join(dataset_image_dir, f"{os.path.splitext(file)[0]}.npy")
#                         np.save(image_npy_path, image_rgb)
#                         print(f"Saved image to {image_npy_path} with shape {image_rgb.shape}.")
#                     else:
#                         print(f"Warning: Could not read image at {image_path}.")

#                     # Tương tự cho nhãn
#                     label_path = os.path.join(label_dir, file)  # Giả sử nhãn có cùng tên file
#                     if os.path.exists(label_path):
#                         label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # Đọc như ảnh xám
#                         if label is not None:
#                             # Resize nhãn về kích thước (512, 512)
#                             label = cv2.resize(label, (512, 512), interpolation=cv2.INTER_NEAREST)
#                             label_npy_path = os.path.join(dataset_label_dir, f"{os.path.splitext(file)[0]}.npy")
#                             np.save(label_npy_path, label)
#                             print(f"Saved label to {label_npy_path} with shape {label.shape}.")
#                         else:
#                             print(f"Warning: Could not read label at {label_path}.")
#                     else:
#                         print(f"Warning: Label not found for {file}.")

# if __name__ == '__main__':
#     images_dirs = [
#         '/home/s12gb1/aima/RetinalVesselSemiSeg/SkinSeg/preprocessing/chase_db1/images',
#         '/home/s12gb1/aima/RetinalVesselSemiSeg/SkinSeg/preprocessing/drive/images',
#         '/home/s12gb1/aima/RetinalVesselSemiSeg/SkinSeg/preprocessing/stare/images'
#     ]
#     labels_dirs = [
#         '/home/s12gb1/aima/RetinalVesselSemiSeg/SkinSeg/preprocessing/chase_db1/labels',
#         '/home/s12gb1/aima/RetinalVesselSemiSeg/SkinSeg/preprocessing/drive/labels',
#         '/home/s12gb1/aima/RetinalVesselSemiSeg/SkinSeg/preprocessing/stare/labels'
#     ]
#     output_npy_dir = '/home/s12gb1/aima/RetinalVesselSemiSeg/SkinSeg/data_processed'
    
#     save_images_and_labels_as_npy(images_dirs, labels_dirs, output_npy_dir)
import numpy as np
import os
import cv2

def save_images_and_labels_as_npy(images_dirs, labels_dirs, output_npy_dir):
    os.makedirs(output_npy_dir, exist_ok=True)  # Tạo thư mục nếu chưa tồn tại

    for dataset_name, image_dir, label_dir in zip(['chase_db1', 'drive', 'stare'], images_dirs, labels_dirs):
        dataset_image_dir = os.path.join(output_npy_dir, dataset_name, 'images')
        dataset_label_dir = os.path.join(output_npy_dir, dataset_name, 'labels')
        
        os.makedirs(dataset_image_dir, exist_ok=True)
        os.makedirs(dataset_label_dir, exist_ok=True)

        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.endswith('.png'):
                    image_path = os.path.join(root, file)
                    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                    if image is not None:
                        image_npy_path = os.path.join(dataset_image_dir, f"{os.path.splitext(file)[0]}.npy")
                        np.save(image_npy_path, image)
                        print(f"Saved image to {image_npy_path}.")
                    else:
                        print(f"Warning: Could not read image at {image_path}.")

                    # Tương tự cho nhãn
                    label_path = os.path.join(label_dir, file)  # Giả sử nhãn có cùng tên file
                    if os.path.exists(label_path):
                        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # Đọc như ảnh xám
                        if label is not None:
                            label_npy_path = os.path.join(dataset_label_dir, f"{os.path.splitext(file)[0]}.npy")
                            np.save(label_npy_path, label)
                            print(f"Saved label to {label_npy_path}.")
                        else:
                            print(f"Warning: Could not read label at {label_path}.")
                    else:
                        print(f"Warning: Label not found for {file}.")

if __name__ == '__main__':
    images_dirs = [
        '/home/s12gb1/aima/RetinalVesselSemiSeg/SkinSeg/preprocessing/chase_db1/images',
        '/home/s12gb1/aima/RetinalVesselSemiSeg/SkinSeg/preprocessing/drive/images',
        '/home/s12gb1/aima/RetinalVesselSemiSeg/SkinSeg/preprocessing/stare/images'
    ]
    labels_dirs = [
        '/home/s12gb1/aima/RetinalVesselSemiSeg/SkinSeg/preprocessing/chase_db1/labels',
        '/home/s12gb1/aima/RetinalVesselSemiSeg/SkinSeg/preprocessing/drive/labels',
        '/home/s12gb1/aima/RetinalVesselSemiSeg/SkinSeg/preprocessing/stare/labels'
    ]
    output_npy_dir = '/home/s12gb1/aima/RetinalVesselSemiSeg/SkinSeg/data_processed'
    
    save_images_and_labels_as_npy(images_dirs, labels_dirs, output_npy_dir)