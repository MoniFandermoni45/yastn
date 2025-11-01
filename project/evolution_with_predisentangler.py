import yastn
import yastn.tn.fpeps as peps
import numpy as np
#import matplotlib.pyplot as plt
from tqdm import tqdm

from yastn.tn.fpeps.gates import gate_nn_Ising, gate_local_field

# Basic (const) parameters
J = -1
h = 5e-4
g = 2.9
beta = 1.643
db = 0.001

geometry = peps.CheckerboardLattice()

# load operators:
opt = yastn.operators.Spin12(sym='dense')

ZZ_gate = gate_nn_Ising(J, db/2, opt.I(), opt.z())
Z_gate  = gate_local_field(h, db/2, opt.I(), opt.z())
X_gate  = gate_local_field(g, db/2, opt.I(), opt.x())

gates = peps.gates.distribute(
    geometry=geometry,
    gates_nn=[ZZ_gate],
    gates_local=[Z_gate, X_gate],
    symmetrize=True
)


# initialize the system in the product state (infinite temperature)
ancilla_psi = peps.product_peps(geometry=geometry, vectors=opt.I())


def get_truncation_errors(t, db, D, method='NN+'):
    '''
    Get array of truncation errors during imaginary time evolution.
    beta   : total time of evolution (imaginary)
    db     : imagiinary time step
    method : method used (for now only NN+)
    '''
    num_steps = int(beta/db)
    opts_svd = {'D_total': D}

    env = peps.EnvNTU(ancilla_psi, which=method)
    infoss = []
    errors = []

    for _ in tqdm(range(num_steps)):
        infos = peps.my_evolution_step(env, gates=gates, opts_svd=opts_svd)
        infoss.append(infos)
        errors.append(peps.accumulated_truncation_error(infoss))
    return errors

def main():

    truncation_error6 = get_truncation_errors(beta, db, D=6)
    truncation_error5 = get_truncation_errors(beta, db, D=5)
    truncation_error4 = get_truncation_errors(beta, db, D=4)

    np.savez('data/errors_ising_001_NN+_withPredisentangler.npz', 
            trunc6= truncation_error6,
            trunc5= truncation_error5,
            trunc4= truncation_error4,
            )

    print('done')

if __name__ == "__main__":
    #main()

    #sym = ancilla_psi[(0,0)].config.sym
    #print(dir(sym))
    main()