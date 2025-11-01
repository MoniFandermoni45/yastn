from ... import tensordot, vdot, svd_with_truncation, YastnError, Tensor
from ._peps import Peps2Layers
from ._gates_auxiliary import Gate, gate_from_mpo
from ..mps import MpsMpoOBC

import yastn.tn.fpeps as peps

from typing import NamedTuple
from itertools import pairwise
import yastn
import numpy as np

# 1. Decompose correctly the tensors A and B
def decompose_A_and_B(env, bond):

    psi = env.psi

    if isinstance(psi, Peps2Layers):
        psi = psi.ket  # to make it work with CtmEnv

    if bond is None: # if bond is not specified, take all bonds
        bonds = psi.bonds()
    else:
        bonds = [bond]

    for bond in bonds:
        dirn = psi.nn_bond_dirn(bond) # take the direction of the bond
        s0, s1 = bond # if l_ordered else bond[::-1] # implement later

        tensor_A = psi[s0]
        tensor_B = psi[s1]

        if dirn == 'h' or 'lr':  # Horizontal gate, "lr" ordered

            #Raxis = 0 meaning specified axes go to the front
            # we want to split the last axis before we do the QR
            tensor_A = tensor_A.unfuse_legs(axes=-1) # t l b r s a <- what we assume
            tensor_B = tensor_B.unfuse_legs(axes=-1) # t l b r s a

            # perform QR:

            # we can immedietely specify the Q axis, we keep ancillas always at the end
            Q0d, R0d = tmpA.qr(axes=((0, 1, 2, 4), (3, 5)), sQ=-1, Qaxis=3)  # t l b rr s @ rr r a
            Q1d, R1d = tmpB.qr(axes=((0, 2, 3, 4), (1, 5)), sQ=1, Qaxis=0, Raxis=1)  # t b r s @ l ll a

            #r0d = R0d.unfuse_legs(axes=2) # rr r s a
            #r0d = r0d.swap_gate(axes=(1, 3)) # swap_gate r and a

            #r1d = R1d.unfuse_legs(axes=1) # l s' a' ll
            #r1d = r1d.swap_gate(axes=(2, 3)) # swap_gate a' and ll

            #print('apply iter')
            return 

            # we include the section to introduce the EAT procedure

            r0d, r1d, diff, num_of_iter = predisentangler_iter(r0d, r1d, D_total, max_iter, tol)
            #print('new r0d:', r0d.get_shape())
            #print('new r1d:', r1d.get_shape())

            r0d = r0d.swap_gate(axes=(1, 3)) # swap_gate r and a
            R0d = r0d.fuse_legs(axes=(0, 1, (2, 3)))

            r1d = r1d.swap_gate(axes=(2, 3))
            R1d = r1d.fuse_legs(axes=(0, (1, 2), 3))

            tmpA = yastn.tensordot(Q0d, R0d, axes=(3, 0)) # t l b r sa

            tmpB = yastn.tensordot(Q1d, R1d, axes=(0, 2)) # t b r l sa
            tmpB = tmpB.transpose(axes=(0, 3, 1, 2, 4))   # t l b r sa

        else: # dirn == 'v':  # Vertical gate, "tb" ordered

            Q0d, R0d = tmpA.qr(axes=((0, 1, 3), (2, 4)), sQ=-1)  # t l r bb @ bb b sa

            Q1d, R1d = tmpB.qr(axes=((1, 2, 3), (0, 4)), sQ=1, Qaxis=0, Raxis=-1)  # tt l b r @ t sa tt

            r0d = R0d.unfuse_legs(axes=2) # bb b s a
            r0d = r0d.swap_gate(axes=(1, 3)) # swap_gate b and a

            r1d = R1d.unfuse_legs(axes=1) # t s' a' tt
            r1d = r1d.swap_gate(axes=(2, 3)) # swap_gate a' and tt

            r0d, r1d, diff, num_of_iter = predisentangler_iter(r0d, r1d, D_total, max_iter, tol)

            r0d = r0d.swap_gate(axes=(1, 3)) # swap_gate r and a
            R0d = r0d.fuse_legs(axes=(0, 1, (2, 3)))

            r1d = r1d.swap_gate(axes=(2, 3))
            R1d = r1d.fuse_legs(axes=(0, (1, 2), 3))

            tmpA = yastn.tensordot(Q0d, R0d, axes=(3, 0)) # t l r b sa
            tmpA = tmpA.transpose(axes=(0, 1, 3, 2, 4))   # t l b r sa

            tmpB = yastn.tensordot(Q1d, R1d, axes=(0, 2)) # l b r t sa
            tmpB = tmpB.transpose(axes=(3, 0, 1, 2, 4))   # t l b r sa


        psi[s0] = tmpA
        psi[s1] = tmpB

    return diff, num_of_iter