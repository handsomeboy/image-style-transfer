import numpy as np
import tensorflow as tf
import skimage.io 

from style_transfer.ext.tf_vgg import vgg19, utils

class Transfer:

  def __init__(self, style, content, width = 240, height = 240, initial = None):
    # Desired size of output image
    self.width = width
    self.height = height
   
    # Read in style and content images and then resize.
    self.style = utils.load_image2(style, self.width, self.height)
    self.content = utils.load_image2(content, self.width, self.height)

    # Use a random starting synthetic image unless an initial is provided.
    self.synthetic = initial if initial else np.random.rand(self.width,                                                                      self.height, 3)

    new_image_shape = (1, self.width, self.height, 3)
    self.style = self.style.reshape(new_image_shape)
    self.content = self.content.reshape(new_image_shape)
    self.synthetic = self.synthetic.reshape(new_image_shape)
    
    self.sess = tf.Session()

    # Create the "VGG19" convolutional neural net.
    self.image = tf.placeholder("float", [1, self.width, self.height, 3])
    self.vgg = vgg19.Vgg19()
    self.vgg.build(self.image)

    # Get the feature maps for the style and content we want.
    self.target_style = self.get_style_features(self.style)
    self.target_content = self.get_content_features(self.content)

  def get_content_features(self, image):
    return self.sess.run(self.vgg.conv4_2, feed_dict= { self.image : image})

  def get_content_loss(self, image):
    image_content = self.get_content_features(image)
    
    loss = 0.0
    # Iterate over each filter, i.
    for i in range(self.target_content.shape[3]):
      # Iterate over each position, j = (x,y)
      for x in range(self.target_content.shape[2]):
        for y in range(self.target_content.shape[1]):
          F = image_content[(0, x, y, i)]
          P = self.target_content[(0, x, y, i)]
          
          loss += (F - P) ** 2.0

    return loss / 2.0

  def get_style_features(self, image):
    style = []

    feed = { self.image : image }

    style.append(self.sess.run(self.vgg.conv1_1, feed_dict=feed))
    style.append(self.sess.run(self.vgg.conv2_1, feed_dict=feed))
    style.append(self.sess.run(self.vgg.conv3_1, feed_dict=feed))
    style.append(self.sess.run(self.vgg.conv4_1, feed_dict=feed))
    style.append(self.sess.run(self.vgg.conv5_1, feed_dict=feed))
    
    return style

  def open_image(self, image_path):
    image = utils.load_image2(image_path, self.width, self.height)
    return image.reshape((1, self.width, self.height, 3,))

