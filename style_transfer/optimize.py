import numpy as np
import copy
from scipy.optimize import fmin_bfgs, fmin_l_bfgs_b

class SGD:
  default_params = {
      'type' : 'sgd',
      'step_size' : 1,
      'iters' : 10,
      'gamma' : 0,
      'eps' : 1e-6,
      'name' : 'SGD',
      'init_display' : lambda *args: None,
      'update_display' : lambda *args: None,
      'save' : lambda *args: None
    }

  required_params = ['theta', 'dJdTheta', 'J']

  def __init__(self, params):
    '''
      params:

        name           : the name used on plots created and files saved
        type           : the type of stochastic gradient descent to use
        step_size      : size of the update step
        iterations     : how many iterations of SGD to perform 
        gamma          : used in momentum and nesterov variations of SGD
        eps            : epsilon value used in adadelta
        theta          : the parameter that is being updated
        dJdTheta       : the gradient of the loss function with respect to theta (returns (grad, loss))
        J              : the loss function
        init_display   : a function that is called to initialize the display
        update_display : a function that given params displays the optimization problem in some way
        save           : a function that is passed params and saves the results somehow
    '''
      
    for param in SGD.required_params:
      if param not in params:
        raise Exception('Missing required parameter' + str(param))
    
    self.params = {}
    self.params.update(SGD.default_params)
    self.params.update(params)
    
    for k,v in self.params.iteritems():
      print(str(k) + " = " + str(v))


  def optimize(self):
    try:
      params = copy.copy(self.params) 
      params['loss'] = []
      
      update = 0
      grad_hist = 0
      update_hist = 0

      gradient, loss = (0,0)
      params['init_display'](params)
      for i in range(1, params['iters'] + 1):
        # stochastic gradient descent
        if params['type'] == 'sgd':
          grad, loss = params['dJdTheta'](params['theta'])
          update = -params['step_size'] * grad
        
        elif params['type'] == 'momentum':
          # Momentum was implemented following this blog:
          # http://ruder.io/optimizing-gradient-descent/index.html#fn:7
          grad, loss = params['dJdTheta'](params['theta'])
          update = params['gamma'] * update - params['step_size'] * grad

        elif params['type'] == 'nesterov':
          # Nesterov was implemented following this blog:
          #http://ruder.io/optimizing-gradient-descent/index.html#fn:7
          loss = params['J'](params['theta'])
          grad, _ = params['dJdTheta'](params['theta']  + params['gamma'] * update)
          update = params['gamma'] * update - (params['step_size'] * grad)

        elif params['type'] == 'adagrad':
          # Adagrad was ipmlemented following this example:
          # https://xcorr.net/2014/01/23/adagrad-eliminating-learning-rates-in-stochastic-gradient-descent/
          grad, loss = params['dJdTheta'](params['theta']) 
          
          # Acuumulate the gradient history
          if np.nonzero(grad_hist):
            grad_hist = params['gamma'] * grad_hist\
                        + (1.0 - params['gamma']) * np.square(grad)
          else:
            grad_hist = np.square(grad)

          adjusted_grad = np.divide(grad, np.sqrt(grad_hist + params['eps']))
          update = -params['step_size'] * adjusted_grad
        
        elif params['type'] == 'adadelta':
          # Adadelta was implemented followeing this paper:
          # https://arxiv.org/pdf/1212.5701.pdf
          grad, loss = params['dJdTheta'](params['theta'])
          grad_hist = params['gamma'] * grad_hist + (1.0 - params['gamma']) * np.square(grad)
          update = -np.multiply(np.divide(
                                          np.sqrt(update_hist + params['eps']), 
                                          np.sqrt(grad_hist + params['eps'])), 
                                          grad)
          update_hist = params['gamma'] * update_hist + (1.0 - params['gamma']) * np.square(update)

        params['theta'] += update
        params['loss'].append(loss)
        params['iter'] = i
        
        params['update_display'](params)
      
      return params['save'](params)
    
    except KeyboardInterrupt:
      return params['save'](params)

  def lgbfs(self):
    print('-------------')
    print('in lgbfs')
    params = copy.copy(self.params)

    func = params['J']
    x0 = params['theta']
    fprime = params['dJdTheta']

    print('func    {}'.format(func))
    print('x0      {}'.format(x0))
    print('x0      {}'.format(x0.shape))
    print('x0 type {}'.format(type(x0)))
    print('x0 type {}'.format(x0.dtype))    # should be float64
    print('fprime  {}'.format(fprime))
    x, f, d = fmin_l_bfgs_b(func, x0, fprime=fprime)

    return x

  # implementation of limited-memory BFGS
  # def lgbfs(self):
  #   print('-------------')
  #   print('in lgbfs')
  #   params = copy.copy(self.params)

  #   x = params['theta']               # initial guess
  #   B = params['dJdThetaHess'](x)     # an approximate hessian matrix

  #   grad_x = params['dJdTheta'](x)

  #   print('about to start iters')
  #   # TODO: check for convergence rather than running an arbitrary number of layers
  #   for i in range(1, params['iters']):
  #     # obtain a direction p by solving Bp = -grad(f(x))
  #     direction = -np.linalg.inv(B) * grad_x

  #     # perform a one-dimensional optimization to find an acceptable
  #     # stepsize in the direction found in the first step
  #     best_step = -1
  #     min_loss = float('inf')
  #     for step in np.arange(0.0, 10.0, .1):
  #       loss = loss(x + step * direction)
  #       if loss < min_loss:
  #         best_step = step

  #     # step s = step*p and update x = x + s
  #     delta = best_step * direction
  #     x += delta

  #     new_grad_x = params['dJdTheta'](x)
  #     y = new_grad_x - grad_x

  #     grad_x = new_grad_x

  #     B += (np.dot(y, np.transpose(y)) / np.dot(np.transpose(y), delta)) - \
  #          ((B @ delta @ np.transpose(delta) @ B) / (np.transpose(delta) @ B @ delta))

  #   return x



