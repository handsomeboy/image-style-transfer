# Image Style Transfer
Bailey Miller and Lily Xu

Final project for CS 89 at Dartmouth, Computational Aspects of Digital Photography.

For our project, we take a stylized painting or patterned image and map its style to an unrelated photograph using a convolutional neural network (CNN).


## Dependencies

- numpy
- tensorflow
- matplotlib
- scipy.optimize (for L-BFGS)
- [libLBFGS](https://github.com/chokkan/liblbfgs) (for L-BFGS)


## Usage

Execute

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
