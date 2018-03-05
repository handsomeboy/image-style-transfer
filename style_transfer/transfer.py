import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import skimage.io
import tensorflow as tf

from ext.tf_vgg import vgg19, utils


# length to pause when displaying plots
PAUSE_LEN = 0.2

class Transfer:

  def __init__(self, style, content, width = 240, height = 240, initial = None):
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

    # Get feature repesentation of content and style
    new_image_shape = (1, self.width, self.height, 3)
    self.style = self.vgg.toBGR(style.reshape(new_image_shape))
    self.content = self.vgg.toBGR(content.reshape(new_image_shape))
    self.synthetic = self.vgg.toBGR(self.synthetic.reshape(new_image_shape))

    # Get the feature maps for the style and content we want
    self.target_style = self.get_style_features(self.style)
    self.target_content = self.get_content_features(self.content)


  #############################################################################
  # content representation
  #############################################################################

  def get_content_features(self, image):
    return self.sess.run(self.vgg.conv4_2, feed_dict={self.image : image})[0]


  def get_content_loss(self, image):
    image_content = self.get_content_features(image)
    target_content = self.target_content
    loss = 0
    for i in range(image_content.shape[2]):
      F_minus_P = np.squeeze(image_content[:,:,i] - self.target_content[:,:,i])
      loss += 0.5 * np.square(F_minus_P)
    return np.mean(loss)


  def get_content_loss_function(self):
    F_minus_P = self.vgg.conv4_2 - tf.constant(self.target_content)
    F_minus_P_2 = 0.5 * tf.square(F_minus_P)
    return tf.reduce_sum(F_minus_P_2)


  def get_content_loss_gradient(self, image):
    loss = self.get_content_loss_function()
    gr = tf.gradients(loss, self.image)
    return self.sess.run(gr, {self.image : image})[0]


  #############################################################################
  # style representation
  #############################################################################

  def get_style_features(self, image):
    style = []

    feed = {self.image : image}

    style.append(self.sess.run(self.vgg.conv1_1, feed_dict=feed)[0])
    style.append(self.sess.run(self.vgg.conv2_1, feed_dict=feed)[0])
    style.append(self.sess.run(self.vgg.conv3_1, feed_dict=feed)[0])
    style.append(self.sess.run(self.vgg.conv4_1, feed_dict=feed)[0])
    style.append(self.sess.run(self.vgg.conv5_1, feed_dict=feed)[0])

    return style


  def get_gram_matrix(self, features):
    style = features

    # vectorize feature maps
    style_v = []
    for l in range(5):
      features = []
      num_feature = style[l].shape[2]
      for i in range(num_feature):
        features.append(style[l][:,:,i].flatten())
      style_v.append(features)

    # Gram matrix for each layer
    num_layer = 5
    gram = []
    for l in range(num_layer):
      num_feature = style[l].shape[2]
      gram.append(np.ndarray((num_feature, num_feature), dtype=float))
      # gram.append(tf.placeholder("float", [num_feature, num_feature]))
      for i in range(num_feature):
        # vectorize feature map
        for j in range(num_feature):
          # vectorize feature maps
          # compute inner product of vectorized feature maps
          gram[l][i,j] = np.inner(style_v[l][i], style_v[l][j])

    return gram


    # TODO: update this function with other formula
    def get_style_loss(self, image):
      image_style = self.get_style_features(image)
      target_style = self.target_style
      loss = 0
      num_feature = image_style.shape[2]
      for i in range(num_feature):
        f_minus_p = np.squeeze(image_style[:,:,i] - self.target_style[:,:,i])
        loss += 0.5 * np.square(f_minus_p)

      return np.mean(loss)


  # style representation
  # loss_style = sum(weighting_l * E_l)
  def get_style_loss_function(self, generated_image):
    A = self.get_gram_matrix(self.target_style)
    G = self.get_gram_matrix(self.get_style_features(generated_image))

    # w_l = 0.2 for conv1_1, 2_1, 3_1, 4_1, and 5_1
    # w_l = 0 on all others
    w_l = 0.2
    num_layer = 5

    loss = 0
    for l in range(num_layer):
      M_l = A[l].shape[0] * A[l].shape[1]  # feature map size (number of features: h x w)
      
      # E_l = 1 / (4 * N_l^2 * M_l^2) * sum( (G_l[i,j] - A_l[i,j])**2 )       
      # TODO: do I want to use np.squeeze() here? after subtracting
      # TODO: do i use tf.reduce_sum instead of np.sum?
      E_l = (4 * num_layer**2  * M_l **2) * (np.sum(np.square(G[l] - A[l])))
      loss += w_l * E_l
    
    return loss


  def get_style_loss_gradient(self, generated_image):
    loss = self.get_style_loss_function(generated_image)

    num_layer = 5

    # F = self.target_style
    # TODO: don't know if this is right to compute matrix F
    F = []
    for l in range(num_layer):
      F.append(np.mean(self.target_style[l], 2))
    A = self.get_gram_matrix(self.target_style)
    G = self.get_gram_matrix(self.get_style_features(generated_image))

    losses = []

    for l in range(num_layer):
      print('-----')
      print('>>   ON LAYER {}'.format(l))
      print("F: {}".format(F[l].shape))
      print("A: {}".format(A[l].shape))
      print("G: {}".format(G[l].shape))
      w = A[l].shape[0]
      h = A[l].shape[1]
      loss = np.ndarray((w, h), dtype=float)


      print('-----------------')
      print(A[l].shape)
      N_l = A[l].shape[0]
      M_l = F[l].shape[0] * F[l].shape[1]

      print('width height {}, {}'.format(w, h))

      for i in range(w):
        for j in range(h):
          # print('searching {}, {} with loss {}'.format(i, j, F[l][i,j]))
          # TODO: is this how i get i and j?
          if F[l][i, j] < 0:
            loss[i, j] = 0
          else:
            loss[i, j] = (1 / (N_l**2 * M_l**2)) * F[l][i,j] * (G[l][j,i] - A[l][j,i])
            # (1 / (N_l**2 * M_l**2)) * F[l].transpose() * (G[l] - A[l])

      losses.append(loss)

    w_l = 0.2
    loss = w_l * sum(losses)  # sum of all 5 arrays in list

    gr = tf.gradients(loss, self.image)
    return self.sess.run(gr, {self.image : generated_image})[0]


  #############################################################################
  # execute
  #############################################################################

  def transfer_only_content(self, step_size = 10.0, iters = 100, img = True, out_dir = "."):
    synthetic = copy.copy(self.synthetic)

    plt.title("Content Transfer")
    plt.ion()
    im = plt.imshow(np.clip(self.vgg.toRGB(synthetic)[0], 0, 1))

    loss = []
    for i in range(iters):
      syn_gradient = self.get_content_loss_gradient(synthetic)
      synthetic -= step_size * syn_gradient
      loss.append(self.get_content_loss(synthetic))
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
    plt.ion()
    im = plt.imshow(np.clip(self.vgg.toRGB(synthetic)[0], 0, 1))

    loss = []
    for i in range(iters):
      syn_gradient = self.get_style_loss_gradient(synthetic)
      synthetic -= step_size * syn_gradient
      loss.append(self.get_style_loss(synthetic))
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
  # unused?
  #############################################################################

  def set_initial_img(self, image):
    self.synthetic = self.vgg.toBGR(image)


  def open_image(self, image_path):
    image = utils.load_image2(image_path, self.width, self.height)
    return image.reshape((1, self.width, self.height, 3,))
