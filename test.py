from utils import load_image, make_dir
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import sys
import glob
import random

def test(ckpt_file, img_path, output_directory):

    make_dir(output_directory)
    
    gen_net = load_model(ckpt_file)
    img = load_image(img_path)

    actual_image = load_image(img_path.replace('degraded', 'correct'))
    img_batch = (np.expand_dims(img, axis=0) - 127.5) / 127.5
    
    img_out = np.squeeze(gen_net.predict(img_batch))
    img_out = img_out * 127.5 + 127.5
    
    img_side_by_side = np.concatenate((img, actual_image, img_out), axis=1)
    img_form = Image.fromarray(img_side_by_side.astype(np.uint8))
    img_form.save(output_directory+'/'+img_path.split('/')[-1])

if __name__ == '__main__':

    best_model = 'generator_checkpoints/generator_5300.h5'
    output_directory = '../output_images'
    test_images = glob.glob('content/data/degraded/test/*/*.jpg')

    random_images = random.sample(test_images, 10)

    [test(best_model, img_path, output_directory) for img_path in random_images]