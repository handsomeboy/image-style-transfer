import numpy as np
import random

noise = np.ndarray((2, 2, 4), dtype=float)

for x in range(2):
	for y in range(2):
	    for c in range(4):
	        noise[x, y, c] = random.uniform(0.0, 1.0)

print('matrix value ' + str(noise))

print('matrix vec ' + str(noise[:,:,:].flatten()))
