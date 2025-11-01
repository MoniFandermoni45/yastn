#from ... import tensordot, vdot, svd_with_truncation, YastnError, Tensor
#from ._peps import Peps2Layers
#from ._gates_auxiliary import Gate, gate_from_mpo
#from ..mps import MpsMpoOBC

#TODO: we need to correct the place where we have 'h' or 'lr' to and add parenthees

import yastn.tn.fpeps as peps

from typing import NamedTuple
from itertools import pairwise
import yastn
import numpy as np

# 1. Decompose correctly the tensors A and B
def decompose_A_and_B(env, bond):
    '''
    Decompose the A and B w.r.t given bond.
    returns: (Q0d, R0d, Q1d, R1d)
    convention:
        - for horizontal bond (left -> right):
        Q0d: t l b rr s
        R0d: rr r a

        Q1d: t ll b r s
        R1d: l ll a

        - for vertical bond (top -> down)
        Q0d: t l bb r s
        R0d: bb b a

        Q1d: tt l b r s
        R1d: t tt a
    '''

    psi = env.psi

    if isinstance(psi, peps.Peps2Layers):
        psi = psi.ket  # to make it work with CtmEnv

    if bond is None: # if bond is not specified, take all bonds
        bonds = psi.bonds()
    else:
        bonds = [bond]

    for bond in bonds:
        dirn = psi.nn_bond_dirn(bond) # take the direction of the bond
        print(dirn)
        s0, s1 = bond # if l_ordered else bond[::-1] # implement later

        tensor_A = psi[s0]
        tensor_B = psi[s1]


        #Raxis = 0 meaning specified axes go to the front
        # we want to split the last axis before we do the QR
        tensor_A = tensor_A.unfuse_legs(axes=4) # t l b r s a <- what we assume
        tensor_B = tensor_B.unfuse_legs(axes=4) # t l b r s a


        if dirn == ('h' or 'lr'):  # Horizontal gate, "lr" ordered
            print('doing horizontal bond')

            # perform QR:
            # we can immedietely specify the Q axis, we keep ancillas always at the end
            Q0d, R0d = tensor_A.qr(axes=((0, 1, 2, 4), (3, 5)), sQ=-1, Qaxis=3)  # t l b rr s @ rr r a
            Q1d, R1d = tensor_B.qr(axes=((0, 2, 3, 4), (1, 5)), sQ=1, Qaxis=1, Raxis=1)  # t ll b r s @ l ll a


        else: # dirn == 'v':  # Vertical gate, "tb" ordered

            print('doing vertical bond')

            Q0d, R0d = tensor_A.qr(axes=((0, 1, 3, 4), (2, 5)), sQ=-1, Qaxis=2)  # t l bb r s @ bb b a
            Q1d, R1d = tensor_B.qr(axes=((1, 2, 3, 4), (0, 5)), sQ=1, Qaxis=0, Raxis=1)  # tt l b r s @ t tt a

    return Q0d, R0d, Q1d, R1d

def main():
    '''tests'''

    config = yastn.make_config(sym='dense')

    # create two tensors
    #legs = [
    #    yastn.Leg(config, s=-1)
    #]

    geometry = peps.CheckerboardLattice()
    opt = yastn.operators.Spin12(sym='dense')
    psi = peps.product_peps(geometry=geometry, vectors=opt.I())
    env = peps.EnvNTU(psi, which='NN')

    #print(env.psi[(0,0)].get_shape())
    # try to decompose the last leg
    #print(env.psi[(0,0)].unfuse_legs(axes=-1)) # the phsical has -1: outgoing, which make sense i guess, ancilla is 1 <- ingoing

    # check the first bond
    bond_lr = peps._geometry.Bond(peps._geometry.Site(0,0), peps._geometry.Site(1,0))

    Q0d, R0d, Q1d, R1d = decompose_A_and_B(env, bond_lr)
    print('Q0d:', Q0d.get_shape())
    print('R0d:', R0d.get_shape())
    print('Q1d:', Q1d.get_shape())
    print('R1d:', R1d.get_shape())

if __name__ == '__main__':
    main()