import numpy as np

x = np.arange(9.0)

y = np.split(x, 3)
print(y)

z = np.reshape(x, (3, 3))
print(z)

w = np.split(z, 3, axis=-1)
print(w)
