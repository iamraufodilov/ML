# load library
import numpy as np
import matplotlib.pyplot as plt

#create data
np.random.seed(0)
x = np.random.rand(100,1)
y = 2 + 3 * x + np.random.rand(100,1)

#plot the data
plt.scatter(x, y, s=10)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
