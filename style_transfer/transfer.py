import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import skimage.io
import tensorflow as tf
import pdb

from ext.tf_vgg import vgg19, utils


# length to pause when displaying plots
PAUSE_LEN = 0.2

class Transfer:

  def __init__(self, style, content, width = 240, height = 240, initial = None):
    self.num_layer = 5

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


  # not executed
  def get_style_features_var(self, image):
    style = []

    feed = {self.image : image}

    style.append(self.vgg.conv1_1)
    style.append(self.vgg.conv2_1)
    style.append(self.vgg.conv3_1)
    style.append(self.vgg.conv4_1)
    style.append(self.vgg.conv5_1)

    return style


  def get_gram_matrix(self, features):
    # Gram matrix for each layer
    gram = []
    for l in range(self.num_layer):
      num_feature = self.sess.run(tf.shape(features[l])[3])
      print("layer {} with {} features".format(l, num_feature))
#      gram.append(np.ndarray((num_feature, num_feature), dtype=float))
      #gram.append(((num_feature, num_feature), dtype = tf.float32))
      gram.append(tf.Variable(tf.zeros([num_feature, num_feature], dtype=tf.float32), dtype=tf.float32))
      # gram.append(tf.placeholder("float", [num_feature, num_feature]))
      n = 0
      for i in range(num_feature):
        for j in range(num_feature):
          if n % 100 == 0:
            print(n)
          # compute inner product of vectorized feature maps
          X = features[l][0,:,:,i]
          Y = features[l][0,:,:,j]
          gram[l][i,j].assign(tf.reduce_sum(tf.multiply(X, Y)))
          n+=1

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


  def get_style_loss_function(self, generated_image):
    # A = self.get_gram_matrix(self.target_style)
    A = self.get_gram_matrix(self.get_style_features_var(self.style))
    G = self.get_gram_matrix(self.get_style_features_var(generated_image))

    # test gram matrix
    G_run = self.sess.run(G[0], {self.image: generated_image})
    print(G_run)

    # A = self.sess.run(A, {self.image : self.style})

    # E = []
    # for l in range(self.num_layer):
    #   N_l = A[l].shape[0]                   # number of features
    #   M_l = A[l].shape[0] * A[l].shape[1]   # feature map size (number of features: h x w)
    #   G_minus_A = tf.constant(G[l]) - tf.constant(A[l])
    #   G_minus_A_2 = 1 / (4 * N_l^2 * M_l^2) * tf.square(G_minus_A)
    #   E.append(tf.reduce_sum(G_minus_A_2))

    #   print('E of layer {} is {}'.format(l, E[-1]))

    # # get weighted sum, which is average
    # return tf.reduce_mean(E)


  def get_style_loss_gradient(self, generated_image):
    loss = self.get_style_loss_function(generated_image)
    print('-----')
    print('get_style_loss_gradient')
    print(loss)
    gr = tf.gradients(loss, self.image)
    print(gr)
    return self.sess.run(gr, {self.image : generated_image})[0]


    # loss = self.get_style_loss_function(generated_image)

    # gr = tf.gradients(loss[0], self.image)
    # print('image')
    # print(self.image.shape)
    # print(gr)
    # print(loss[0])
    # total = self.sess.run(gr, {self.image : generated_image})[0]

    # self.num_layer = 5
    # for l in range(1, self.num_layer):
    #   gr = tf.gradients(loss[l], self.image)
    #   total += self.sess.run(gr, {self.image : generated_image})[0]

    # print('-----')
    # print('get_style_loss_gradient')
    # print(total.shape)
    # return total

  # style representation
  # loss_style = sum(weighting_l * E_l)
  # def get_style_loss_function(self, generated_image):
  #   A = self.get_gram_matrix(self.target_style)
  #   G = self.get_gram_matrix(self.get_style_features(generated_image))

  #   # w_l = 0.2 for conv1_1, 2_1, 3_1, 4_1, and 5_1
  #   # w_l = 0 on all others
  #   w_l = 0.2
  #   self.num_layer = 5

  #   loss = 0
  #   for l in range(self.num_layer):
  #     M_l = A[l].shape[0] * A[l].shape[1]  # feature map size (number of features: h x w)
      
  #     # E_l = 1 / (4 * N_l^2 * M_l^2) * sum( (G_l[i,j] - A_l[i,j])**2 )       
  #     # TODO: do I want to use np.squeeze() here? after subtracting
  #     # TODO: do i use tf.reduce_sum instead of np.sum?
  #     E_l = (4 * self.num_layer**2  * M_l **2) * (np.sum(np.square(G[l] - A[l])))
  #     loss += w_l * E_l
    
  #   return loss


  # def get_style_loss_gradient(self, generated_image):
  #   print('-------------')
  #   print('get_style_loss_gradient')
  #   loss_fxn = self.get_style_loss_function(generated_image)
  #   print(loss_fxn)
  #   loss = tf.convert_to_tensor(loss_fxn)
  #   print(loss)
  #   print(loss.shape)
  #   gr = tf.gradients(loss, self.image)
  #   print(gr)
  #   return self.sess.run(gr, {self.image : generated_image})[0]


    # self.num_layer = 5

    # # F = self.target_style
    # # TODO: don't know if this is right to compute matrix F
    # F = []
    # for l in range(self.num_layer):
    #   r = self.target_style[l].shape[2]
    #   c = self.target_style[l].shape[0]**2
    #   F.append(np.reshape(self.target_style[l], (r, c)))
    # A = self.get_gram_matrix(self.target_style)
    # G = self.get_gram_matrix(self.get_style_features(generated_image))

    # gr_losses = []

    # for l in range(self.num_layer):
    #   print('-----')
    #   print('>>   ON LAYER {}'.format(l))
    #   print("F: {}".format(F[l].shape))
    #   print("A: {}".format(A[l].shape))
    #   print("G: {}".format(G[l].shape))
    #   w = A[l].shape[0]
    #   h = A[l].shape[1]
    #   loss = np.ndarray((w, h), dtype=float)

    #   print('-----------------')
    #   print(A[l].shape)
    #   N_l = A[l].shape[0]
    #   M_l = F[l].shape[0] * F[l].shape[1]

    #   print('width height {}, {}'.format(w, h))

    #   # if F[l]_i,j < 0, then deriv should be 0
    #   F_use = np.clip(F[l].transpose(), 0, None)
    #   loss = (1 / (N_l**2 * M_l**2)) * F_use.dot(G[l] - A[l])

    #   gr_losses.append(loss)
    #   # print('loss on layer {} is {}'.format(l, gr_losses[-1]))

    # # w_l = 0.2
    # # loss = w_l * sum(gr_losses)  # sum of all 5 arrays in list
    # # print('total loss is ', loss)

    # # try returning only layer 4, with size that matches
    # return gr_losses[3]
    # # return gr_losses

    # # losses = tf.convert_to_tensor(gr_losses, dtype=tf.float32)
    # # # losses = tf.reduce_sum(gr_losses, 0)
    # # gr = tf.gradients(losses, self.image)

    # # print(losses)
    # # print(gr)
    # # return self.sess.run(gr, {self.image : generated_image})[0]


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
      print('-------')
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
      for x in syn_gradient:
        print(x.shape)
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
