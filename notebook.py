import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x = np.arange(0.01, 1, 0.01)
y = np.arange(0.01, 1, 0.01)
z = []
for i in x:
    for j in y:
        z.append((-np.log(x) * y) + (-np.log(1 - x), 1 - y))
z = np.reshape(z,(len(z),1))
# z = (-np.log(x) * y) + (-np.log(1 - x), 1 - y)
fig = plt.figure()
ax=Axes3D(fig)
ax.plot(x, z, y)
plt.show()
