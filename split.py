import os
import numpy as np
import random

def create_folds(proceeded_folder='./proceeded_data', num_folds=5):
    folds_dir = proceeded_folder
    os.makedirs(folds_dir, exist_ok=True)

    image_files = os.listdir(os.path.join(proceeded_folder, 'images'))
    label_files = os.listdir(os.path.join(proceeded_folder, 'labels'))

    assert all(img[:-4] == lbl[:-4] for img, lbl in zip(sorted(image_files), sorted(label_files))), "Image and label names do not match."

    file_names = [img[:-4] for img in image_files]

    random.shuffle(file_names)

    folds = np.array_split(file_names, num_folds)

    for i in range(num_folds):
        with open(os.path.join(folds_dir, f'fold{i + 1}.txt'), 'w') as f:
            for name in folds[i]:
                f.write(f"{name}.npy\n")

if __name__ == '__main__':
    create_folds(proceeded_folder='./proceeded_data/chase_db1')
    create_folds(proceeded_folder='./proceeded_data/drive')
    create_folds(proceeded_folder='./proceeded_data/stare')

