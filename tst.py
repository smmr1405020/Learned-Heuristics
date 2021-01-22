import numpy as np

a = np.reshape(np.indices((64,64)), newshape=(2, -1)).T
print(a)