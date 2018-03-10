import numpy as np
import copy
from scipy.optimize import fmin_l_bfgs_b

class SGD:
  default_params = {
      'type' : 'sgd',
      'step_size' : 1,
      'iters' : 10,
      'gamma' : 0,
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


  def optimize(self):
    params = copy.copy(self.params) 
    params['loss'] = []
    update = 0

    grad_hist = 0

    gradient, loss = (0,0)
    params['init_display'](params)
    for i in range(1, params['iters']):
      # stochastic gradient descent
      if params['type'] == 'sgd':
        grad, loss = params['dJdTheta'](params['theta'])
        update = params['step_size'] * grad
      
      # stochastic gradient descent with momentum
      elif params['type'] == 'momentum':
        grad, loss = params['dJdTheta'](params['theta'])
        update = params['gamma'] * update + params['step_size'] * grad

      # Nesterov's accelerated gradient descent
      elif params['type'] == 'nesterov':
        loss = params['J'](params['theta'])
        grad, _ = params['dJdTheta'](params['theta'] - params['gamma'] * update)
        update = params['gamma'] * update + params['step_size'] * grad

      # adaptive gradient descent
      elif params['type'] == 'adagrad':
        grad, loss = params['dJdTheta'](params['theta'] - params['gamma'] * update) 
        grad_hist += np.square(grad)
        update = params['step_size'] * np.divide(grad, 1e-6 + np.sqrt(grad_hist))

      params['theta'] -= update
      params['loss'].append(loss)
      params['iter'] = i
      
      params['update_display'](params)

    return params['save'](params)


  # limited-memory BFGS
  # factr:  1e12 for low accuracy
  #         1e7 for moderate accuracy
  #         10.0 for extremely high accuracy
  def optimize_lbfgs(self, factr=1e15):
    params = copy.copy(self.params)
    params['loss'] = []

    func = params['J']
    x0 = params['theta']
    fprime = params['dJdTheta']

    x, f, d = fmin_l_bfgs_b(func, x0, fprime=fprime, factr=factr)
    params['loss'].append(d)
    params['theta'] = x

    return params['save'](params)
