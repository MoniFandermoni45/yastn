import numpy as np
import matplotlib.pyplot as plt

diff   = np.load('data/bond_dim_diff_D=6_002.npy') - 12
errors = np.load('data/errors_from_bond_dim_D=6_002_.npy')

plt.semilogy(diff, errors, '*-')
plt.xlabel("'Dr' - D*r")
plt.ylabel('Truncation error')
plt.title('Bond dependance D=6')

plt.savefig('visualization/bond_dependance_D=6.svg')
plt.show()