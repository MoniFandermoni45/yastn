
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
db = 0.01

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
#ancilla_psi = peps.product_peps(geometry=geometry, vectors=opt.I())


#env = peps.EnvNTU(ancilla_psi, which='NN')
#env = peps.EnvCTM(ancilla_psi, init='eye')

#opts_svd = {'D_total': 5} # works for D_total = 1



def main():
#    info = peps.evolution_step_(env, gates=gates, opts_svd=opts_svd)
    #print('---------')
    #info = peps.evolution_step_(env, gates=gates, opts_svd=opts_svd)

    #opts_svd = {'D_total': 5}
    #info = env.ctmrg_(opts_svd=opts_svd, max_sweeps=200, )

    #print(type(env.psi))
    #siteA = env.psi.sites()[0]
    #siteB = env.psi.sites()[1]
    #print(siteA, siteB)
    #print(env[siteA].tl.to_dense())

    # how does the swap gate works:
    my_config = yastn.make_config(sym='dense')
    leg = yastn.Leg(my_config, s=1, t=(), D=(2,))
    my_tensor = yastn.rand(my_config, legs=[leg, leg.conj(), leg])
    print(my_tensor.s)

if __name__ == '__main__':
    main()

#with PyCallGraph(output=GraphvizOutput(output_file='call_graph_depth=2.png')):
#    evolution_step()

