import os
import numpy as np
import skimage
import skimage.io
from transfer import Transfer

WIDTH = 244
HEIGHT = 244

DATA_INPUT = "data/input/"
DATA_OUTPUT = "data/output/"

style_path = os.path.join(DATA_INPUT, "style/vangogh.jpg")
content_path = os.path.join(DATA_INPUT, "content/baker.jpg")

synthetic_name = os.path.splitext(style_path)[0].split("/")[-1]
synthetic_name += "_on_"
synthetic_name += os.path.splitext(content_path)[0].split("/")[-1]

transfer = Transfer(style_path, content_path, WIDTH, HEIGHT)

# test content transfer
# transfer.transfer_only_content(img = True, step_size = 0.001, iters = 100, out_dir = DATA_OUTPUT)

# test style transfer
transfer.transfer_only_style(step_size = 10.0, iters = 100, out_dir = DATA_OUTPUT)


