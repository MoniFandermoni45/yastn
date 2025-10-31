import numpy as np
import matplotlib.pyplot as plt

data = np.load('data/errors_ising_01_NN+.npz')

errors6 = data['trunc6']
errors5 = data['trunc5']
errors4 = data['trunc4']

plt.plot(errors6)
plt.plot(errors5)
plt.plot(errors4)
plt.grid(linewidth=.2)
plt.show()