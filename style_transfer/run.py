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
#style_path = os.path.join(DATA_INPUT, "style/rocks.jpg")
content_path = os.path.join(DATA_INPUT, "content/baker.jpg")

synthetic_name = os.path.splitext(style_path)[0].split("/")[-1]
synthetic_name += "_on_"
synthetic_name += os.path.splitext(content_path)[0].split("/")[-1]

transfer = Transfer(style_path, content_path, WIDTH, HEIGHT,
                    initial = None,
                    content_layers = ["conv4_2"])

# test content transfer
transfer.transfer_only_content(out_dir = DATA_OUTPUT, params = {
                                'type' : 'nesterov',
                                'step_size' : 100,
                                'iters' : 30,
                                'gamma' : 0.9,
                                'gamma_normal' : 20,
                                'add_noise' : False,
                                'eps' : 1,
                                'beta' : 0.0,
                              })

# test style transfer
#transfer.transfer_only_style(step_size = 1.0e-9, iters = 100, out_dir = DATA_OUTPUT)

