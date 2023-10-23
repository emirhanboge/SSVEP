import os
import subprocess
import random
import time

from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import mean_absolute_error

def load_images_and_labels_AFAD_FULL(root_folder="../AFAD-FULL"):
    images_and_labels = []

    if not os.path.exists(root_folder):
        print(f"The folder {root_folder} does not exist.")
        return None

    for age_label in tqdm(os.listdir(root_folder), desc="Loading Images"):
        age_folder = os.path.join(root_folder, age_label)

        if os.path.isdir(age_folder) and age_label.isdigit():

            for dirpath, _, filenames in os.walk(age_folder):

                for filename in filenames:
                    if filename.lower().endswith(".jpg"):
                        full_image_path = os.path.join(dirpath, filename)
                        images_and_labels.append((full_image_path, int(age_label)))
    return images_and_labels

def load_images_and_labels_UTKFace(root_folder="../../UTKFace"):
    images_and_labels = []

    if not os.path.exists(root_folder):
        print(f"The folder {root_folder} does not exist.")
        return None

    for filename in tqdm(os.listdir(root_folder), desc="Loading Images"):
        if filename.lower().endswith(".jpg"):
            full_image_path = os.path.join(root_folder, filename)
            age_label = filename.split("_")[0]
            if int(age_label) > 10 and int(age_label) < 100:
                images_and_labels.append((full_image_path, int(age_label)))
    return images_and_labels

def randomly_choose_n_images(images_and_labels, n):
    random.shuffle(images_and_labels)
    return images_and_labels[:n]

def load_flicker_images(root_folder="../../UTKFlicker"):
    images = []

    if not os.path.exists(root_folder):
        print(f"The folder {root_folder} does not exist.")
        return None

    for filename in tqdm(os.listdir(root_folder), desc="Loading Images"):
        if filename.lower().endswith(".jpg"):
            full_image_path = os.path.join(root_folder, filename)
            images.append(full_image_path)
    return images

def make_prediction(image_path, label):
    try:
        result = subprocess.run(['./predict.sh', image_path], capture_output=True)
        age = int(result.stdout.strip().decode('utf-8'))
        label = int(label)
        return label, age
    except Exception as e:
        print(f"Error while processing {image_path}: {e}")
        return label, None

def save_activations(image_path):
    try:
        result = subprocess.run(['./predict.sh', image_path], capture_output=True)
    except Exception as e:
        print(f"Error while processing {image_path}: {e}")
        return label, None

def sort_by_2digits(images):
    sorted_images = []
    for image in images:
        if "updown" in image:
            index = int(image.split("_")[2].split(".")[0])
        else:
            index = int(image.split("_")[3].split(".")[0])
        sorted_images.append((image, index))

    sorted_images.sort(key=lambda x: x[1])
    return [image for image, _ in sorted_images]

def evaluate_model(image_folder, output_filepath):
    images_and_labels = load_images_and_labels_AFAD_FULL(image_folder)
    images_and_labels = randomly_choose_n_images(images_and_labels, 1000)
    print(f"Loaded {len(images_and_labels)} images.")

    y_true = []
    y_pred = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(executor.map(make_prediction, [img for img, _ in images_and_labels], [lbl for _, lbl in images_and_labels]), total=len(images_and_labels), desc="Processing Images"))

    with open(output_filepath, 'w') as output_file:
        for label, age in results:
            if age is not None:
                y_true.append(label)
                y_pred.append(age)

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)

        output_file.write(f"MORPH2 MAE: 5.68\n")
        output_file.write(f"MAE: {mae}\n")
        output_file.write(f"N: {len(y_true)}\n")
        output_file.write("TRUE | PRED\n")
        for true, pred in zip(y_true, y_pred):
            output_file.write(f"{true} {pred}\n")

def shuffle_or_sort_images(images, task):
    if task == "shuffle": # shuffles the images order
        random.shuffle(images)
    else: # sorts the images by the index
        images = sort_by_2digits(images)
    return images

"""
image_folder = "../../Pruning_Experiments/UTKFlicker_1/"
images = load_flicker_images(image_folder)
images = shuffle_or_sort_images(images, "sort")
for image in tqdm(images):
    save_activations(image)
"""

image_folder = "../../Pruning_Experiments/"
folders_under_image_folder = os.listdir(image_folder)
folders_under_image_folder = [folder for folder in folders_under_image_folder if folder.startswith("UTKFlicker")]
folders_under_image_folder.sort()

for folder in folders_under_image_folder:
    images = load_flicker_images(os.path.join(image_folder, folder))
    images = shuffle_or_sort_images(images, "sort")
    for image in tqdm(images):
        save_activations(image)

