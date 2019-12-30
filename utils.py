import os
import numpy as np
from PIL import Image

def load_image(image_path):
    return(np.array(Image.open(image_path)))

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
