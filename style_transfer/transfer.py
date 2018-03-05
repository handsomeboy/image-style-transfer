import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import skimage.io 
import tensorflow as tf

from style_transfer.ext.tf_vgg import vgg19, utils

class Transfer:

  def __init__(self, style, content, width = 240, height = 240, initial = None):
    # Desired size of output image
    self.width = width
    self.height = height
   
    # Read in style and content images and then resize.
    self.style = utils.load_image2(style, self.width, self.height)
    self.content = utils.load_image2(content, self.width, self.height)

    self.sess = tf.Session()

    # Create the "VGG19" convolutional neural net.
    self.image = tf.placeholder("float", [1, self.width, self.height, 3])
    self.vgg = vgg19.Vgg19()
    self.vgg.build(self.image)

    if initial == None:
      rand_noise = np.random.rand(self.width,self.height)
      rand_noise = rand_noise.reshape(1,self.width,self.height,1)
      white_noise = np.concatenate((rand_noise,rand_noise,rand_noise), axis=3)
      self.synthetic = white_noise
    else:
      self.synthetic = initial

    new_image_shape = (1, self.width, self.height, 3)
    self.style = self.vgg.toBGR(self.style.reshape(new_image_shape))
    self.content = self.vgg.toBGR(self.content.reshape(new_image_shape))
    self.synthetic = self.vgg.toBGR(self.synthetic.reshape(new_image_shape))

    # Get the feature maps for the style and content we want.
    self.target_style = self.get_style_features(self.style)
    self.target_content = self.get_content_features(self.content)
    
  def get_content_features(self, image):
    return self.sess.run(self.vgg.conv4_2, feed_dict= { self.image : image})[0]

  def get_content_loss(self, image):
    image_content = self.get_content_features(image)
    target_content = self.target_content
    loss = 0
    for i in range(image_content.shape[2]):
      F_MINUS_P = np.squeeze(image_content[:,:,i] - self.target_content[:,:,i])
      loss += 0.5 * np.square(F_MINUS_P)
    return np.mean(loss)

  def get_content_loss_function(self):
    F_MINUS_P = self.vgg.conv4_2 - tf.constant(self.target_content)
    F_MINUS_P_2 = 0.5 * tf.square(F_MINUS_P)
    return tf.reduce_sum(F_MINUS_P_2)

  def get_content_loss_gradient(self, image): 
    loss = self.get_content_loss_function()
    gr = tf.gradients(loss, self.image)
    return self.sess.run(gr, { self.image : image })[0]
    
  def get_style_features(self, image):
    style = []

    feed = { self.image : image }

    style.append(self.sess.run(self.vgg.conv1_1, feed_dict=feed)[0])
    style.append(self.sess.run(self.vgg.conv2_1, feed_dict=feed)[0])
    style.append(self.sess.run(self.vgg.conv3_1, feed_dict=feed)[0])
    style.append(self.sess.run(self.vgg.conv4_1, feed_dict=feed)[0])
    style.append(self.sess.run(self.vgg.conv5_1, feed_dict=feed)[0])
    
    return style
  
  def set_initial_img(self, image):
    self.synthetic = self.vgg.toBGR(image)

  def transfer_only_content(self, step_size = 10.0, iters = 100, img = True, out_dir = "."):
    synthetic = copy.copy(self.synthetic)

    plt.title("Content Transfer")
    plt.ion()
    im = plt.imshow(np.clip(self.vgg.toRGB(synthetic)[0], 0, 1))

    loss = []
    for i in range(iters):
      syn_gradient = self.get_content_loss_gradient(synthetic)
      synthetic -= step_size*syn_gradient
      loss.append(self.get_content_loss(synthetic))
      im.set_data(np.clip(self.vgg.toRGB(synthetic)[0], 0, 1)) 
      plt.pause(0.2)
      print(loss[-1])

    out = self.vgg.toRGB(synthetic)
    out = np.clip(out, 0, 1)
    skimage.io.imsave(os.path.join(out_dir, "content_transfer.jpg"), out[0])
  
    plt.plot(loss)
    plt.savefig(os.path.join(out_dir, "content_loss.jpg"))

    return out
    
  def open_image(self, image_path):
    image = utils.load_image2(image_path, self.width, self.height)
    return image.reshape((1, self.width, self.height, 3,))

