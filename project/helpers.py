import yastn
import yastn.tn.fpeps as peps
import numpy as np
from yastn.tn.fpeps._my_evolution import my_evolution_step
from yastn.tn.fpeps._evolution import evolution_step_, my_evolution_step
from yastn.tn.fpeps.gates import gate_nn_Ising, gate_local_field

def get_Ising_gates(parameters):

    h  = parameters['h']
    J  = parameters['J']
    db = parameters['db']
    g  = parameters['g']

    opt = yastn.operators.Spin12(sym='dense')
    geometry = peps.CheckerboardLattice()

    ZZ_gate = gate_nn_Ising(J, db/2, opt.I(), opt.z())
    Z_gate  = gate_local_field(h, db/2, opt.I(), opt.z())
    X_gate  = gate_local_field(g, db/2, opt.I(), opt.x())

    gates = peps.gates.distribute(
        geometry=geometry,
        gates_nn=[ZZ_gate],
        gates_local=[Z_gate, X_gate],
        symmetrize=True
    )

    return gates

def get_infinite_temperature_state():

    geometry = peps.CheckerboardLattice()
    opt = yastn.operators.Spin12(sym='dense')

    ancilla_psi = peps.product_peps(geometry=geometry, vectors=opt.I())

    return ancilla_psi

