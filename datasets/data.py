import os
import numpy as np
from skimage.io import imread
from skimage.transfrom import resize

def read_data(data_dir, divided_grid=32):
