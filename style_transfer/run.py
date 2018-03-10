import os
from transfer import Transfer
import time

WIDTH = 256
HEIGHT = 256

DATA_INPUT = 'data/input/'
DATA_OUTPUT = 'data/output/'

# style_path = os.path.join(DATA_INPUT, 'style/vangogh.jpg')
# style_path = os.path.join(DATA_INPUT, 'style/rocks.jpg')
style_path = os.path.join(DATA_INPUT, 'style/checkerboard.jpg')
content_path = os.path.join(DATA_INPUT, 'content/baker.jpg')

synthetic_name = os.path.splitext(style_path)[0].split('/')[-1]
synthetic_name += '_on_'
synthetic_name += os.path.splitext(content_path)[0].split('/')[-1]

transfer = Transfer(style_path, content_path, WIDTH, HEIGHT,
                    initial = None,
                    content_layers = ['conv4_2'])

transfer.set_initial_img(content_path)

start = time.time()

#style = transfer.open_image(style_path)
content = transfer.open_image(content_path)

#skimage.io.imsave(os.path.join(DATA_OUTPUT, "style.jpg"), style[0])
#skimage.io.imsave(os.path.join(DATA_OUTPUT, "content.jpg"), content[0])

# test content transfer
#transfer.transfer_only_content(out_dir = DATA_OUTPUT, params = {
#                                'type' : 'nesterov',
#                                'step_size' : 100,
#                                'iters' : 30,
#                                'gamma' : 0.9,
#                              })

# test style transfer
# transfer.transfer_only_style(step_size = 1.0e-9, iters = 2, out_dir = DATA_OUTPUT)


# NOTE: paper suggests alpha/beta = 1e-3

# test entire transfer
# transfer.transfer_style_to_image(out_dir = DATA_OUTPUT,
#                                  alpha = 1,       # content weighting
#                                  beta = 1e3,      # style weighting
#                                  params = {
#                                    'type' : 'adagrad',
#                                    'step_size' : 20,
#                                    'iters' : 1,
#                                    'gamma' : 2
#                                  })

# test transfer using L-BFGS optimization
transfer.transfer_style_to_image_lbfgs(out_dir = DATA_OUTPUT,
                                 alpha = 1,       # content weighting
                                 beta = 1e3,      # style weighting
                                 params = { 'type' : 'lbfgs',
                                            'factr' : 1e13})

end = time.time()
print('Total runtime: {} seconds'.format(end - start))
