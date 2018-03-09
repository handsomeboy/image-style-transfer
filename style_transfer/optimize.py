import numpy as np
import copy

class SGD:
  def __init__(self, params):
    '''
      params:

        name : the name used on plots created and files saved.
        type : the type of stochastic gradient descent to use
        step_size : size of the update step
        iters : how many iterations of SGD to perform 
        gamma : used in momentum and nesterov variations of SGD
        theta : the parameter that is being updated
        dJdTheta : the gradient of the loss function with respect to theta  (returns (grad, loss))
        J : the loss function
        init_display : a function that is called to initialize the display.
        update_display :  a function that given params displays the optimization problem in some way.
        save : a function that is passed params and saves the results somehow.
    '''
    self.params = params

  def optimize(self):
    params = copy.copy(self.params) 
    params['loss'] = []
    last_update = 0
    gradient, loss = (0,0)

    params['init_display'](params)
    for i in range(params['iters']):
      if params['type'] == 'sgd':
        grad, loss = params['dJdTheta'](params['theta'])
        update = params['step_size'] * grad
      
      elif params['type'] == 'momentum':
        grad, loss = params['dJdTheta'](params['theta'])
        update = params['gamma'] * last_update + params['step_size'] * grad

      elif params['type'] == 'nesterov':
        loss = params['J'](params['theta'])
        grad, _ = params['dJdTheta'](params['theta'] - params['gamma'] * last_update)
        update = params['gamma'] * last_update + params['step_size'] * grad

      params['theta'] -= update
      last_update = update
      params['loss'].append(loss)
      params['iter'] = i
      
      params['update_display'](params)

    params['save'](params)
