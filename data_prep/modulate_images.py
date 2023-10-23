import os
import random

import numpy as np
import matplotlib.pyplot as plt

import cv2
import imageio

def load_random_images(folder_path, num_images=5):
    all_images = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
    selected_images = random.sample(all_images, num_images)
    return [cv2.imread(os.path.join(folder_path, img)) for img in selected_images]

def sinusoidal(t, frequency=1):
    return 0.45 * np.sin(2 * np.pi * frequency * t) + 0.55

def apply_sin_modulation(image, luminance):
    return np.clip(image * luminance, 0, 255).astype(np.uint8)

def generate_images(image, num_images, frequency=12):
    generated_images = []
    for i in range(num_images):
        t = i / 120
        luminance = sinusoidal(t, frequency)
        modulated_image = apply_sin_modulation(image, luminance)
        generated_images.append(modulated_image)
    return generated_images

def plot_sinusoidal_graph(num_captured_points=3, num_plot_points=100, frequency=12):
    t_values_captured = np.linspace(0, 0.1, num_captured_points)  # 0 to 0.1 seconds for captured images
    t_values_plot = np.linspace(0, 0.1, num_plot_points)  # 0 to 0.1 seconds for plotting
    y_values_captured = [sinusoidal(t, frequency) for t in t_values_captured]
    y_values_plot = [sinusoidal(t, frequency) for t in t_values_plot]

    plt.scatter(t_values_captured, y_values_captured, color='red', label='Image Captured')
    plt.plot(t_values_plot, y_values_plot)
    plt.title('Custom Sinusoidal Function')
    plt.xlabel('Time (sec)')
    plt.ylabel('Relative Luminance (0 - 1)')
    plt.legend()
    plt.savefig('sinusoidal.png')
    plt.show()


if __name__ == "__main__":
    folder_path = '../UTKFace'
    destination_path = '../UTKFlicker'
    num_images_to_select = 1
    num_generated_images = 120

    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    random_images = load_random_images(folder_path, num_images_to_select)

    print(f'Generating {num_generated_images} images for each of the {num_images_to_select} randomly selected images')

    for i, image in enumerate(random_images):
        generated_images = generate_images(image, num_generated_images)
        for j, img in enumerate(generated_images):
            cv2.imwrite(f'{destination_path}/Image_{j+1}.jpg', img)

