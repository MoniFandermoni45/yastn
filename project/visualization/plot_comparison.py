import numpy as np
import matplotlib.pyplot as plt

data_onlyNTU = np.load('data/errors_ising_01_NN+_onlyNTU_correct.npz')
my_version_predisentangler = np.load('data/errors_ising_01_NN+_withPredisentangler_myVersion_version_1_correct.npz')
Yintai_version_predisentangler = np.load('data/errors_ising_01_NN+_withPredisentangler_Yintai_version.npz')


errors_onlyNTU_6 = data_onlyNTU['trunc6']
errors_onlyNTU_5 = data_onlyNTU['trunc5']
errors_onlyNTU_4 = data_onlyNTU['trunc4']

my_version_predisentangler_6 = my_version_predisentangler['trunc6']
my_version_predisentangler_5 = my_version_predisentangler['trunc5']
my_version_predisentangler_4 = my_version_predisentangler['trunc4']

Yintai_version_predisentangler_6 = Yintai_version_predisentangler['trunc6']
Yintai_version_predisentangler_5 = Yintai_version_predisentangler['trunc5']
Yintai_version_predisentangler_4 = Yintai_version_predisentangler['trunc4']

beta = np.linspace(0, 1.643, len(errors_onlyNTU_6))

plt.plot(beta, errors_onlyNTU_6, 'g:',label='NN+ pre=0 dt=0.01 D_total=6')
plt.plot(beta, errors_onlyNTU_5, 'r:',label='NN+ pre=0 dt=0.01 D_total=5')
plt.plot(beta, errors_onlyNTU_4, 'b:',label='NN+ pre=0 dt=0.01 D_total=4')

plt.plot(beta, my_version_predisentangler_6, 'g-',label='NN+ pre=1 dt=0.01 D_total=6')
plt.plot(beta, my_version_predisentangler_5, 'r-',label='NN+ pre=1 dt=0.01 D_total=5')
plt.plot(beta, my_version_predisentangler_4, 'b-',label='NN+ pre=1 dt=0.01 D_total=4')

plt.plot(beta, Yintai_version_predisentangler_6, 'g*',label='NN+ pre=1 dt=0.01 D_total=6 Y')
plt.plot(beta, Yintai_version_predisentangler_5, 'r*',label='NN+ pre=1 dt=0.01 D_total=5 Y')
plt.plot(beta, Yintai_version_predisentangler_4, 'b*',label='NN+ pre=1 dt=0.01 D_total=4 Y')

plt.grid(linewidth=0.2)
plt.xlabel(r'$\beta$')
plt.ylabel(r'$\rho$')
plt.title(r'Accumulated evolution error')
plt.legend()

plt.savefig('visualization/NTU_vs_my_version_method1_vs_Yintai_version.svg')
plt.show()


