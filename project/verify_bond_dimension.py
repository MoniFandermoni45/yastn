import yastn
import matplotlib.pyplot as plt
import numpy as np
import yastn.tn.fpeps as peps
from yastn.tn.fpeps._my_evolution import my_evolution_step
from helpers import get_infinite_temperature_state, get_Ising_gates
from parameters import parameters # import model parameters
from tqdm import tqdm

gates = get_Ising_gates(parameters=parameters)

psi = get_infinite_temperature_state()

def get_truncation_errors(t, db, D, method='NN+'):
    '''
    Get array of truncation errors during imaginary time evolution.
    beta   : total time of evolution (imaginary)
    db     : imagiinary time step
    method : method used (for now only NN+)
    '''
    num_steps = int(t/db)
    opts_svd = {'D_total': D}

    # initialize the system in the product state (infinite temperature)
    ancilla_psi = get_infinite_temperature_state()
    env = peps.EnvNTU(ancilla_psi, which=method)

    infoss = []
    errors = []

    for _ in tqdm(range(num_steps)):
        infos, r0dr1d, R0d, R1d = my_evolution_step(env, gates, opts_svd=opts_svd, methodType='method_1')
        infoss.append(infos)
        errors.append(peps.accumulated_truncation_error(infoss))
    return errors, r0dr1d, R0d, R1d

# at first we calculate the number of singular values
def get_singular_values(matrix):
    _, s, _ = yastn.svd(matrix)
    singular_values = np.diag(s.to_numpy())
    return singular_values

def svd_approx(A: yastn.Tensor, n):
    """
    Return the best rank-n approximation of matrix A
    using the first n singular values from the SVD.
    """
    A = A.to_numpy()

    # Full SVD
    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    # Keep only first n components
    U_n = U[:, :n]
    S_n = S[:n]
    Vt_n = Vt[:n, :]

    # Reconstruct rank-n approximation
    return U_n @ np.diag(S_n) @ Vt_n

def main():
    beta = parameters['beta']
    db = parameters['db']
    _, r0dr1d, R0d, R1d =  get_truncation_errors(t=beta, db=db, D=6, method='NN+')

    # at first we need to fuse
    r0dr1d = r0dr1d.fuse_legs(axes=((0,1), (2,3)))
    singular_values_main = get_singular_values(r0dr1d)

    r0dr1d_truncated = yastn.tensordot(R0d, R1d, axes=((1), (0))) # rr a ll a'
    r0dr1d_truncated = r0dr1d_truncated.fuse_legs(axes=((0,1), (2,3)))
    singular_values_truncated = get_singular_values(r0dr1d_truncated)

    print(singular_values_main)
    print()
    print(singular_values_truncated)

    # print('Number of all singular values:', len(singular_values_main))
    # print('Number of truncated singular values:', len(singular_values_truncated))

    errors = []
    for i in range(12, len(singular_values_main)+1):
        err = np.linalg.norm(r0dr1d_truncated.to_numpy() - svd_approx(r0dr1d, i), ord='fro')
        errors.append(err)
    
    # np.save('data/bond_dim_diff_D=6_002.npy', np.array(list(range(12, len(singular_values_main)+1))))
    # np.save('data/errors_from_bond_dim_D=6_002_.npy', errors)
    np.save('data/main_singular_values_D=6_003.npy', singular_values_main)

    
    # plt.semilogy(list(range(10, 41)), errors, '*')
    # plt.show()



if __name__ == '__main__':
    main()