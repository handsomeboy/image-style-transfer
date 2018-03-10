# Image Style Transfer
Bailey Miller and Lily Xu

Final project for CS 89 at Dartmouth, Computational Aspects of Digital Photography.

For our project, we take a stylized painting or patterned image and map its style to an unrelated photograph using a convolutional neural network (CNN).

We were given permission by Professor Jarosz to use Python rather than C for our project. 


## Dependencies

- numpy
- tensorflow
- matplotlib
- scipy.optimize (for L-BFGS)
- [libLBFGS](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html)


## Usage

In `run.py`, update the variables at top to select the style and content images desired, select the type of style transfer, and the optimization initialization. 

Types of style transfer are:

- `rand2style`: white noise image to style transfer
- `rand2content`: white noise image to content transfer
- `style2img`: style transfer using gradient descent
- `style2imgLBFGS`: style transfer using L-BFGS

For style transfer, if `rand = True`, the optimization will be initialized using a white noise image. Otherwise, the optimization will be initialized using the content image.

For advanced tuning, modify the parameters passed to the Transfer class.

To run the Image Style Transfer program, execute

```sh
python run.py
```



## References

This project is based off the paper
> Gatys, L.A., Ecker, A.S., Bethge, M. 
> Image Style Transfer Using Convolutional Neural Networks, CVPR 2016.

To implement our work, we used VGG19, a trained convolutional neural network from [Github user machrisaa](https://github.com/machrisaa/tensorflow-vgg) based off
> Simonyan K., Zisserman A. Very Deep Convolutional Networks for Large-Scale Image Recognition, CVPR 2014.

Additional references include:
> Gatys, L.A., Ecker, A.S., Bethge, M.
> Texture Synthesis Using Convolutional Neural
Networks, NIPS 2015.

> Nocedal, J. Updating Quasi-Newton Matrices with Limited Storage. Mathematics of Computation 1980.

> Zeiler, M. ADADELTA: An Adaptive Learning Rate Method. Computing Research Repository, 2012.

> Ruder, Sebastian. "An overview of gradient descent optimization algorithms." Ruder.io, 26 Jan 2016.

> Mineault, Patrick. "Adagrad eliminating learning rates in stochastic gradient descent." Xcorr, 23 Jan 2014.