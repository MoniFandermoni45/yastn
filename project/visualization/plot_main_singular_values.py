import numpy as np
import matplotlib.pyplot as plt

singular_values = np.load('data/main_singular_values_D=6_003.npy')

plt.semilogy(singular_values, '-*')
plt.title('Singular values of r0dr1d matrix (before truncation)')
plt.savefig('visualization/singular_values_D=6.svg')
plt.show()