import os
import numpy as np
import skimage
import skimage.io
from transfer import Transfer
import time

WIDTH = 244
HEIGHT = 244

DATA_INPUT = "data/input/"
DATA_OUTPUT = "data/output/"

# style_path = os.path.join(DATA_INPUT, "style/vangogh.jpg")
style_path = os.path.join(DATA_INPUT, "style/rocks.jpg")
content_path = os.path.join(DATA_INPUT, "content/baker.jpg")

synthetic_name = os.path.splitext(style_path)[0].split("/")[-1]
synthetic_name += "_on_"
synthetic_name += os.path.splitext(content_path)[0].split("/")[-1]

transfer = Transfer(style_path, content_path, WIDTH, HEIGHT,
                    initial = None,
                    content_layers = ["conv1_2", "conv4_2"])

start = time.time()

# test content transfer
#transfer.transfer_only_content(step_size = 1000, iters = 100, out_dir = DATA_OUTPUT)

# test style transfer

transfer.transfer_only_style(step_size = 3e-9, iters = 100, out_dir = DATA_OUTPUT)

end = time.time()
print('Total runtime: {} seconds'.format(end - start))


