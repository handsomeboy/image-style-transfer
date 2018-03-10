from argparse import *
import os
import numpy as np
import skimage
import skimage.io
from transfer import Transfer
import time

WIDTH = 244
HEIGHT = 244

DATA_INPUT = 'data/input/'
DATA_OUTPUT = 'data/output/'

if __name__ == "__main__":
  # Coose content and style image.
  style_path = os.path.join(DATA_INPUT, 'style/vangogh.jpg')
  content_path = os.path.join(DATA_INPUT, 'content/baker.jpg')

  transfer = Transfer(style_path, content_path, WIDTH, HEIGHT,
                      initial = None)
  transfer.set_initial_img(content_path)

  start = time.time()

  style = transfer.open_image(style_path)
  content = transfer.open_image(content_path)
  
  # Saves a copy of the style and content image (in reshaped form).
  skimage.io.imsave(os.path.join(DATA_OUTPUT, "style.jpg"), style[0])
  skimage.io.imsave(os.path.join(DATA_OUTPUT, "content.jpg"), content[0])

  # Select the style transfer methods to run.
  rand2style = False
  rand2content = True
  style2img = False
  style2imgLBFGS = False

  # Select if you want your initial image to be random or content.
  rand = True

  # test content transfer
  if rand2content:
    transfer.set_random_initial_img()
    transfer.transfer_only_content(out_dir = DATA_OUTPUT, params = {
                                  'type' : 'momentum',
                                  'step_size' : 100,
                                  'iters' : 30,
                                  'gamma' : 0.9,
                                  'eps': 1e-6
                                })

  if rand2style:
    transfer.set_random_initial_img()
    transfer.transfer_only_style(out_dir = DATA_OUTPUT, params = {
                                  'type' : 'adadelta',
                                  'step_size' : 1e-1,
                                  'iters' : 15,
                                  'gamma' : 0.9,
                                  'eps': 1e-6
                                })

  # NOTE: paper suggests alpha/beta = 1e-3

  # test entire transfer
  if style2img:
    if rand:
      transfer.set_random_initial_img()
    else:
      transfer.set_initial_img(content_path)
    
    transfer.transfer_style_to_image(out_dir = DATA_OUTPUT,
                                   alpha = 5e5,       # content weighting
                                   beta = 1,      # style weighting
                                   params = {
                                      'type' : 'momentum',
                                      'step_size' : 1e-6,
                                      'iters' : 100,
                                      'gamma' : 0.9,
                                      'eps' : 1e-6
                                   })
  if style2imgLBFGS:
    transfer.transfer_style_to_image_lbfgs(out_dir = DATA_OUTPUT,
                                   alpha = 1,       # content weighting
                                   beta = 1e3,      # style weighting
                                   params = {
                                     'type' : 'adagrad',
                                     'step_size' : 20,
                                     'iters' : 1,
                                     'gamma' : 2
                                   })
  
  end = time.time()
  print('Total runtime: {} seconds'.format(end - start))
