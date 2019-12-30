from utils import load_image
import glob
import random
import numpy as np

def data_loader(batch_size):
    correct_train_images = sorted(glob.glob('content/data/correct/train/*/*.jpg'))
    random.shuffle(correct_train_images)
    i = 0
    while True:
        correct_list = []
        degraded_list = []
        for index in range(batch_size):
            if i == len(correct_train_images):
                i = 0
                random.shuffle(correct_train_images)
            correct_image = load_image(correct_train_images[i])
            degraded_image = load_image(correct_train_images[i].replace('correct', 'degraded'))
            i += 1
            correct_list.append(correct_image)
            degraded_list.append(degraded_image)
        correct_image_batch = (np.array(correct_list) - 127.5) / 127.5
        degraded_image_batch = (np.array(degraded_list) - 127.5) / 127.5
        yield(correct_image_batch, degraded_image_batch)
