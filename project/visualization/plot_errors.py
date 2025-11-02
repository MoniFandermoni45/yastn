import numpy as np
import matplotlib.pyplot as plt

data = np.load('data/errors_ising_01_NN+.npz')
data_pred = np.load('data/errors_ising_01_NN+_withPredisentangler.npz')
data_pred_001 = np.load('data/errors_ising_001_NN+_withPredisentangler.npz')
data_pred_01_myVersion = np.load('data/errors_ising_01_NN+_withPredisentangler_myVersion.npz')
data_pred_01_myVersion_version_2 = np.load('data/errors_ising_01_NN+_withPredisentangler_myVersion_version_2.npz')


errors6 = data['trunc6']
errors5 = data['trunc5']
errors4 = data['trunc4']

errors6_pred = data_pred['trunc6']
errors5_pred = data_pred['trunc5']
errors4_pred = data_pred['trunc4']

errors6_pred_01_myVersion = data_pred_01_myVersion['trunc6']
errors5_pred_01_myVersion = data_pred_01_myVersion['trunc5']
errors4_pred_01_myVersion = data_pred_01_myVersion['trunc4']

errors6_pred_01_myVersion_version_2 = data_pred_01_myVersion_version_2['trunc6']
errors5_pred_01_myVersion_version_2 = data_pred_01_myVersion_version_2['trunc5']
errors4_pred_01_myVersion_version_2 = data_pred_01_myVersion_version_2['trunc4']

errors6_pred_001 = data_pred_001['trunc6']
errors5_pred_001 = data_pred_001['trunc5']
errors4_pred_001 = data_pred_001['trunc4']

x = np.linspace(0, 1.643, len(errors6))
x2 = np.linspace(0, 1.643, len(errors6_pred_001))

plt.plot(x, errors6, 'g-', label='D_total=6 pre=0')
plt.plot(x, errors5, 'b-', label='D_total=5 pre=0')
plt.plot(x, errors4, 'r-', label='D_total=4 pre=0')

plt.plot(x, errors6_pred, 'g--', label='D_total=6 pre=1')
plt.plot(x, errors5_pred, 'b--', label='D_total=5 pre=1')
plt.plot(x, errors4_pred, 'r--', label='D_total=4 pre=1')

plt.plot(x, errors6_pred_01_myVersion, 'g:', label='D_total=6 pre=1 (myVersion)')
plt.plot(x, errors5_pred_01_myVersion, 'b:', label='D_total=5 pre=1 (myVersion)')
plt.plot(x, errors4_pred_01_myVersion, 'r:', label='D_total=4 pre=1 (myVersion)')

plt.plot(x, errors6_pred_01_myVersion_version_2, 'g.', label='D_total=6 pre=1 (myVersion) version 2')
plt.plot(x, errors5_pred_01_myVersion_version_2, 'b.', label='D_total=5 pre=1 (myVersion) version 2')
plt.plot(x, errors4_pred_01_myVersion_version_2, 'r.', label='D_total=4 pre=1 (myVersion) version 2')

#plt.plot(x2, errors6_pred_001, 'g*', label='D_total=6 pre=1 dt=.001')
#plt.plot(x2, errors5_pred_001, 'b*', label='D_total=5 pre=1 dt=.001')
#plt.plot(x2, errors4_pred_001, 'r*', label='D_total=4 pre=1 dt=.001')

plt.grid(linewidth=.2)
plt.xlabel(r'$\beta$')
plt.ylabel(r'$\rho$')
plt.title(r'Accumulated evolution error')
plt.legend()

plt.savefig('visualization/accumulated_evolution_error_NN+_01_myVersion_version_2.svg')
plt.show()
