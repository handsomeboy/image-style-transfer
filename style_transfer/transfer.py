import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import skimage.io
import tensorflow as tf
import pdb

from ext.tf_vgg import vgg19, utils


# length to pause when displaying plots
PAUSE_LEN = 0.01

class Transfer:

  def __init__(self, style, content, width = 240, height = 240, initial = None, 
               content_layers = ["conv4_2"], style_layers = ["conv1_1","conv2_1","conv3_1","conv4_1","conv5_1"]):
    self.content_layers = content_layers
    self.style_layers = style_layers

    # Desired size of output image
    self.width = width
    self.height = height

    self.sess = tf.Session()

    # Create the "VGG19" convolutional neural net
    self.image = tf.placeholder("float", [1, self.width, self.height, 3])
    self.vgg = vgg19.Vgg19()
    self.vgg.build(self.image)

    if not initial:
      rand_noise = np.random.rand(self.width, self.height)
      rand_noise = rand_noise.reshape(1, self.width, self.height,1)
      white_noise = np.concatenate((rand_noise, rand_noise, rand_noise), axis=3)
      self.synthetic = white_noise
    else:
      self.synthetic = initial

    # Read in style and content images and then resize
    style = utils.load_image2(style, self.width, self.height)
    content = utils.load_image2(content, self.width, self.height)

    #Convert content and style to BGR.
    new_image_shape = (1, self.width, self.height, 3)
    self.style = self.vgg.toBGR(style.reshape(new_image_shape))
    self.content = self.vgg.toBGR(content.reshape(new_image_shape))
    self.synthetic = self.vgg.toBGR(self.synthetic.reshape(new_image_shape))

    # Get the feature maps for the style and content we want
    self.target_content = self.get_content_features(self.content)
   
    # Create symbolic gram matrices and target gram matrices for style image.
    self.gram_matrix_functions = self.get_gram_matrices()
    self.target_gram_matrices = [] 
    for G in self.gram_matrix_functions:
      self.target_gram_matrices.append(self.sess.run(G, { self.image : self.style }))
    

  #############################################################################
  # content representation
  #############################################################################

  def get_content_features(self, image):
    content = {}
    for layer in self.content_layers:
      content[layer] = self.sess.run(self.vgg[layer], feed_dict={self.image : image})[0]
    return content 

  def get_content_loss(self, image):
    loss = self.get_content_loss_function()
    return self.sess.run(loss, {self.image : image })

  def get_content_loss_function(self):
    content_layer_loss = []
    for layer in self.content_layers:
      F_minus_P = self.vgg[layer] - tf.constant(self.target_content[layer])
      F_minus_P_2 = 0.5 * tf.square(F_minus_P)
      content_layer_loss.append(tf.reduce_mean(F_minus_P_2))
    return tf.reduce_mean(content_layer_loss)

  def get_content_loss_gradient(self, image):
    loss = self.get_content_loss_function()
    gr = tf.gradients(loss, self.image)
    content_gradient = self.sess.run(gr, {self.image : image})[0]
    content_loss = self.sess.run(loss, { self.image : image})
    return (content_gradient, content_loss)


  #############################################################################
  # style representation
  #############################################################################

  def get_style_features(self):
    style = []
    for layer in self.style_layers:
      style.append(self.vgg[layer])
    return style


  def get_gram_matrices(self):
    features = self.get_style_features()
    gram_matrices = []
    for l in range(len(self.style_layers)):
      num_feature = self.sess.run(tf.shape(features[l])[3])

      # Using "-1" flattens the tensor. So -1 = "M_L**2" in this case.
      A = tf.reshape(features[l], [-1, num_feature])
      gram_matrices.append(tf.matmul(A, A, transpose_a = True))
    return gram_matrices


  def get_style_loss(self, image):
    loss_style = get_style_loss_function(image)
    return self.sess.run(loss, {self.image : image})

  def get_style_loss_function(self, generated_image):
    E = []
    for l in range(len(self.target_gram_matrices)):
      E.append(tf.reduce_mean(tf.square(self.gram_matrix_functions[l] - self.target_gram_matrices[l])))
    return tf.reduce_mean(E)

  def get_style_loss_gradient(self, generated_image):
    loss = self.get_style_loss_function(generated_image)
    gr = tf.gradients(loss, self.image)
    style_gradient = self.sess.run(gr, {self.image : generated_image})[0]
    style_loss = self.sess.run(loss, { self.image : generated_image})
    return (style_gradient, style_loss)

  #############################################################################
  # execute
  #############################################################################

  def transfer_only_content(self, step_size = 10.0, iters = 100, out_dir = "."):
    synthetic = copy.copy(self.synthetic)

    plt.title("Content Transfer")
    plt.ion()
    im = plt.imshow(np.clip(self.vgg.toRGB(synthetic)[0], 0, 1))

    loss = []
    for i in range(iters):

      syn_gradient, syn_loss = self.get_content_loss_gradient(synthetic)
      print('-------')
      synthetic -= step_size * syn_gradient
      loss.append(syn_loss)
      im.set_data(np.clip(self.vgg.toRGB(synthetic)[0], 0, 1))
      plt.pause(PAUSE_LEN)
      print("Loss on iteration {}: {}".format(i, loss[-1]))

    out = self.vgg.toRGB(synthetic)
    out = np.clip(out, 0, 1)
    skimage.io.imsave(os.path.join(out_dir, "content_only_transfer.jpg"), out[0])

    plt.plot(loss)
    plt.savefig(os.path.join(out_dir, "content_loss.jpg"))

    return out


  def transfer_only_style(self, step_size = 10.0, iters = 100, out_dir = "."):
    synthetic = copy.copy(self.synthetic)   # get white noise image

    plt.title("Style Transfer")
    plt.ion()                               # interactive mode on
    im = plt.imshow(np.clip(self.vgg.toRGB(synthetic)[0], 0, 1))

    loss = []
    for i in range(iters):
      (syn_gradient, style_loss) = self.get_style_loss_gradient(synthetic)
      synthetic -= step_size * syn_gradient
      loss.append(style_loss)
      im.set_data(np.clip(self.vgg.toRGB(synthetic)[0], 0, 1))
      plt.pause(PAUSE_LEN)
      print("Loss on iteration {}: {}".format(i, loss[-1]))

    out = self.vgg.toRGB(synthetic)
    out = np.clip(out, 0, 1)
    skimage.io.imsave(os.path.join(out_dir, "style_only_transfer.jpg"), out[0])

    plt.plot(loss)
    plt.savefig(os.path.join(out_dir, "style_loss.jpg"))

    return out


  #############################################################################
  # Utility
  #############################################################################

  def set_initial_img(self, image):
    self.synthetic = self.vgg.toBGR(image)

  def open_image(self, image_path):
    image = utils.load_image2(image_path, self.width, self.height)
    return image.reshape((1, self.width, self.height, 3,))
