import yastn
import yastn.tn.fpeps as peps
import numpy as np
#import matplotlib.pyplot as plt
from tqdm import tqdm

import yastn.tn.fpeps.envs._env_auxlliary as env_aux

from yastn.tn.fpeps.gates import gate_nn_Ising, gate_local_field


geometry = peps.CheckerboardLattice()
opt = yastn.operators.Spin12(sym='Z2')

ancilla_psi = peps.product_peps(geometry=geometry, vectors=opt.I())

def main():
    '''Main thing'''
    
    tensor = env_aux.trivial_peps_tensor(ancilla_psi.config)
    print(tensor)

    my_config = yastn.make_config(sym='Z2')

    my_leg = yastn.Leg(my_config, s=1, t=(0,1), D=(1,1))

    my_tensor = yastn.ones(
        config=my_config,
        legs = [my_leg.conj(), my_leg, my_leg, my_leg.conj(), my_leg],
        n=0
    )
    #my_tensor = my_tensor.add_leg(axis=-1, s=1)
    print('----------')
    print(my_tensor)

    print('----------')
    print(my_tensor.swap_gate(0,1))

    print('----------')
    print(my_tensor.transpose(axes=(1,0,2,3,4)))
    #yastn.ones()


if __name__ == "__main__":
    main()