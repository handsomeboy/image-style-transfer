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
    update = 0

    grad_hist = 0
    step_hist = 0

    gradient, loss = (0,0)
    params['init_display'](params)
    for i in range(1,params['iters']):
      if params['type'] == 'sgd':
        grad, loss = params['dJdTheta'](params['theta'])
        update = params['step_size'] * grad
      
      elif params['type'] == 'momentum':
        grad, loss = params['dJdTheta'](params['theta'])
        update = params['gamma'] * last_update + params['step_size'] * grad

      elif params['type'] == 'nesterov':
        loss = params['J'](params['theta'])
        grad, _ = params['dJdTheta'](params['theta'] - params['gamma'] * update)
        update = params['gamma'] * update + params['step_size'] * grad

      elif params['type'] == 'adagrad':
        grad, loss = params['dJdTheta'](params['theta'] - params['beta'] * update) 
        grad_hist += np.square(grad)
        update = params['step_size'] * np.divide(grad, 1e-6 + np.sqrt(grad_hist))

      elif params['type'] == 'adadelta':
        # http://climin.readthedocs.io/en/latest/adadelta.html
        grad, loss = params['dJdTheta'](params['theta'] - params['beta'] * update)
  
        # Accumulate gradient.
        grad_hist = np.add(np.multiply(params['gamma'], grad_hist), 
                           np.multiply((1.0 - params['gamma']), np.square(grad)))
        
        # delta_theta = sqrt(step_hist / grad_hist) * grad
        delta_theta = params['step_size'] * np.multiply(np.divide(np.sqrt(step_hist + params['eps']),
                                                                  np.sqrt(grad_hist + params['eps'])),
                                                        grad)

        # Accumulate steps
        step_hist = np.add(np.multiply(params['gamma'], step_hist), 
                           np.multiply((1.0 - params['gamma']), np.square(delta_theta)))
        
        update = delta_theta

      sigma_noise = np.sqrt(params['step_size'] / (1 + i) ** params['gamma_normal'])
      noise = np.zeros(update.shape) if params['add_noise'] else np.random.uniform(0,sigma_noise,update.shape)
      params['theta'] -= update + noise
      params['loss'].append(loss)
      params['iter'] = i
      
      params['update_display'](params)

    params['save'](params)
