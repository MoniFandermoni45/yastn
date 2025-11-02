#from ... import tensordot, vdot, svd_with_truncation, YastnError, Tensor
#from ._peps import Peps2Layers
#from ._gates_auxiliary import Gate, gate_from_mpo
#from ..mps import MpsMpoOBC

#TODO: we need to correct the place where we have 'h' or 'lr' to and add parenthees

import yastn.tn.fpeps as peps

from typing import NamedTuple, Union
from itertools import pairwise
import yastn
import numpy as np

from yastn.tn.fpeps.gates import gate_nn_Ising, gate_local_field

# 1. Decompose correctly the tensors A and B
def decompose_A_and_B(env, bond):
    '''
    Decompose the A and B w.r.t given bond.
    returns: (Q0d, R0d, Q1d, R1d)
    convention:

    - for horizontal bond (left -> right): \\
        Q0d: t l b rr s
        R0d: rr r a 

        Q1d: t ll b r s
        R1d: l ll a 

    - for vertical bond (top -> down):
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
        s0, s1 = bond # if l_ordered else bond[::-1] # implement later

        tensor_A = psi[s0]
        tensor_B = psi[s1]


        #Raxis = 0 meaning specified axes go to the front
        # we want to split the last axis before we do the QR
        tensor_A = tensor_A.unfuse_legs(axes=4) # t l b r s a <- what we assume
        tensor_B = tensor_B.unfuse_legs(axes=4) # t l b r s a

        #print('TensorA:', tensor_A.get_shape())


        if dirn == 'h' or  dirn == 'lr':  # Horizontal gate, "lr" ordered

            #print('Doing horizontal bond')

            # perform QR:
            # we can immedietely specify the Q axis, we keep ancillas always at the end
            Q0d, R0d = tensor_A.qr(axes=((0, 1, 2, 4), (3, 5)), sQ=-1, Qaxis=3)  # t l b rr s @ rr r a
            Q1d, R1d = tensor_B.qr(axes=((0, 2, 3, 4), (1, 5)), sQ=1, Qaxis=1, Raxis=1)  # t ll b r s @ l ll a


        else: # dirn == 'v':  # Vertical gate, "tb" ordered

            #print('Doing vertical bond')

            Q0d, R0d = tensor_A.qr(axes=((0, 1, 3, 4), (2, 5)), sQ=1, Qaxis=2)  # t l bb r s @ bb b a
            Q1d, R1d = tensor_B.qr(axes=((1, 2, 3, 4), (0, 5)), sQ=-1, Qaxis=0, Raxis=1)  # tt l b r s @ t tt a
    
    #print('Results:')
    #print('Q0d:', Q0d.get_shape())

    return Q0d, R0d, Q1d, R1d

# 2. Find the metric of the 

# TODO: change the env type,
def my_evolution_step(env: peps.EnvNTU, gates, opts_svd, method='mpo', fix_metric=0,
                    pinv_cutoffs=(1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4),
                    max_iter=100, tol_iter=1e-13, initialization="EAT_SVD"):
    
    psi: Union[peps.Peps, peps.Peps2Layers] = env.psi # experimental
    if isinstance(psi, peps.Peps2Layers):
        psi = psi.ket  # to make it work with CtmEnv

    infos = []

    for gate in gates:
        psi.apply_gate_(gate)

        for s0, s1 in pairwise(gate.sites[-1::-1]):
            env.pre_truncation_((s0, s1)) # for now nothing
        if len(gate.sites) > 2:
            for s0, s1 in pairwise(gate.sites):
                env.pre_truncation_((s0, s1)) # for now nothing

        for s0, s1 in pairwise(gate.sites):
            # here we'll use the predisentangler info = truncate_(env, opts_svd, (s0, s1), fix_metric, pinv_cutoffs, max_iter, tol_iter, initialization)

            # 1. Decompose the tensors at Sites s0 at s1
            D_total = opts_svd['D_total']
            bond = peps._geometry.Bond(s0,s1)
            # print('Doing bond:', bond)
            # print('site0:', psi[s0].get_shape())
            # print('site1:', psi[s1].get_shape())
            Q0d, R0d, Q1d, R1d = decompose_A_and_B(env, bond)

            # 2, Get the metric based on the sites s0, s1
            dirn = psi.nn_bond_dirn(bond) # take the direction of the bond

            # print('Q0d:', Q0d.get_shape())
            # print('Q1d:', Q1d.get_shape())
            # print('R0d:', R0d.get_shape())
            # print('R1d:', R1d.get_shape())
            fgf = env.bond_metric(Q0=Q0d, Q1=Q1d, s0=s0, s1=s1, dirn=dirn)

            # the metric for lr: [rr' ll'] [rr ll]
            # the metric for tb: [bb' tt'] [bb tt]

            # 3. Apply the metric to R0dR1d to get R0dR1d_tilde

            # 4. Put this to modified Yintai's procedure -> we get (pre)disentangler

            # 5. One way to continue is to take simply the svd of g applied to RR
            # 6. Other way is to apply it nevertheless, in such way update the tensors and then put this to standard NTU procedure.

            # print(fgf)

            #diff, num_of_iter = peps.apply_predisentangler(env, bond, D_total=D_total, max_iter=max_iter) # this does not perform the truncation # I guess
            # it only prepair the system to the further evolution ...
            #info = truncate_(env, opts_svd, (s0, s1), max_iter)
            #info = peps.truncate_(env, opts_svd, (s0, s1), fix_metric, pinv_cutoffs, max_iter, tol_iter, initialization)
            #infos.append(info)
        
    return infos


def my_apply_predisentangler(r0d: yastn.Tensor, r1d: yastn.Tensor, D_total, max_iter=400, tol=1e-7):


    r0d, r1d, diff, num_of_iter = my_predisentangler_iter(r0d, r1d, D_total, max_iter, tol)
    # we can update the tensors

    # We need to perform this back contraction:

    r0d = r0d.swap_gate(axes=(1, 3)) # swap_gate r and a
    R0d = r0d.fuse_legs(axes=(0, 1, (2, 3)))

    r1d = r1d.swap_gate(axes=(2, 3))
    R1d = r1d.fuse_legs(axes=(0, (1, 2), 3))

    tmpA = yastn.tensordot(Q0d, R0d, axes=(3, 0)) # t l r b sa
    tmpA = tmpA.transpose(axes=(0, 1, 3, 2, 4))   # t l b r sa

    tmpB = yastn.tensordot(Q1d, R1d, axes=(0, 2)) # l b r t sa
    tmpB = tmpB.transpose(axes=(3, 0, 1, 2, 4))   # t l b r sa


    # we may set it
    # psi[s0] = tmpA
    # psi[s1] = tmpB

    return diff, num_of_iter

def my_predisentangler_iter(r0d:yastn.Tensor, r1d:yastn.Tensor, D_total, max_iter, tol=1e-7):
    '''
    returns: (
    r0d: "left" reduced tensor,
    r1d: "right" reduced tensor,
    diff: difference in the convergence,
    ii: number of iterations
    )
    '''

    ii = 0
    diff = 32767

    #r0dr1d = yastn.tensordot(r0d, r1d, axes=(1, 0)) # contr. r and l
    r0dr1d = yastn.tensordot(r0d, r1d, axes=(1,0)) # rr a ll a' (My version)
    #r0dr1d = yastn.transpose(r0dr1d, axes=(0, 2, 1, 3)) # rr ll a a'

    for ii in range(max_iter):

        #u, s, v = yastn.svd_with_truncation(r0dr1d, axes=((0, 1, 2), (3, 4, 5)), sU=r0d.s[1], D_total=D_total)
        u, s, v = yastn.svd_with_truncation(r0dr1d, axes=((0, 1), (2, 3)), sU=r0d.s[1], D_total=D_total) # U: rr a r   V^dag: l ll a'
        r0dr1d_new = s.broadcast(u, axes=2)

        # we obtain the truncated middle
        r0dr1d_new = yastn.tensordot(r0dr1d_new, v, axes=(2, 0)) # rr a ll a' (My version)

        g = build_predisentangler_g(r0dr1d, r0dr1d_new.conj()) # a a* a' a'* <- Yintai and my version

        # xx s a s' a' yy <- Yintai version
        # my version rr a ll a'
        #r0dr1d_new = yastn.tensordot(r0dr1d, g, axes=((2, 4), (1, 3))) # xx s s' yy a a' Yintai version
        r0dr1d_new = yastn.tensordot(r0dr1d, g, axes=((1, 3), (1, 3))) # rr ll a a' my version

        # Divide onto left and right part (but first transpose after application of the disentangler):
        #r0dr1d_new = r0dr1d_new.transpose(axes=(0, 1, 4, 2, 5, 3)) # xx s a s' a' yy Yintai version
        r0dr1d_new = r0dr1d_new.transpose(axes=(0, 2, 3, 1)) # rr a a' ll 

        # Normalization:
        #r0dr1d_new = r0dr1d_new / (r0dr1d_new.fuse_legs(axes=((0, 1, 2), (3, 4, 5))).norm(p="fro"))
        r0dr1d_new = r0dr1d_new / (r0dr1d_new.fuse_legs(axes=((0, 1), (2, 3))).norm(p="fro"))


        # Calculate the error:
            # For Yintai: r0dr1d: rr s a s' a' ll
            # For me    : r0dr1d: rr a a' ll
            #diff = (r0dr1d.fuse_legs(axes=((0, 1, 2), (3, 4, 5))) - r0dr1d_new.fuse_legs(axes=((0, 1, 2), (3, 4, 5)))).norm(p="fro") / r0dr1d.fuse_legs(axes=((0, 1, 2), (3, 4, 5))).norm(p="fro")
        diff = (r0dr1d.transpose(axes=(0,1,3,2)).fuse_legs(axes=((0, 1), (2, 3))) - r0dr1d_new.fuse_legs(axes=((0, 1), (2, 3)))).norm(p="fro") / r0dr1d.fuse_legs(axes=((0, 1, 2), (3, 4, 5))).norm(p="fro")

        r0dr1d = r0dr1d_new
        r0dr1d = r0dr1d.transpose(axes=(0,1,3,2)) # rr a ll a'
        # we may probably transpose back here to make sure we match the pattern rr a ll a'

        if diff < tol:
            # if the difference is sufficiently small


            # Yintai version: r0dr1d: xx s a s' a' yy
            # My version    : r0dr1d: rr a ll a'
            #u, s, v = yastn.svd_with_truncation(r0dr1d, axes=((0, 1, 2), (3, 4, 5)), sU=r0d.s[1], D_total=r0d.get_shape(axes=1))
            u, s, v = yastn.svd_with_truncation(r0dr1d, axes=((0,1), (2,3)), sU=r0d.s[1], UAxis = 1, D_total=r0d.get_shape(axes=1)) # u: r rr a, v: l ll a'

            # Redestribute the singular values
            s = s.sqrt()
            #r0d = s.broadcast(u, axes=3).transpose(axes=(0, 3, 1, 2))
            r0d = s.broadcast(u, axes=1) # rr r a
            r1d = s.broadcast(v, axes=0) # l ll a'
            break

    return r0d, r1d, diff, ii

def build_predisentangler_g(r0dr1d:yastn.Tensor, r0dr1d_conj:yastn.Tensor):
    # r0dr1d xx s a s' a' yy <- Yintai version (xx==rr, yy==ll, probably)
    # rodr1d rr a ll a' <- my version

    #Eg = yastn.tensordot(r0dr1d, r0dr1d_conj, axes=((0, 1, 3, 5), (0, 1, 3, 5))) # a a' a* a'*  Yintai
    Eg = yastn.tensordot(r0dr1d, r0dr1d_conj, axes=((0,2), (0, 2))) # a a' a* a'*
    u, s, v = Eg.svd(axes=((0, 1), (2, 3)))
    Eg = yastn.tensordot(v.conj(), u.conj(), axes=(0, 2)) # a* a'* a a' (&) # to be verified!
    Eg = Eg.transpose(axes=(0, 2, 1, 3))
    return Eg



def main():
    '''tests'''

    config = yastn.make_config(sym='dense')

    # Basic (const) parameters
    J = -1
    h = 5e-4
    g = 2.9
    beta = 1.643
    db = 0.01


    geometry = peps.CheckerboardLattice()
    opt = yastn.operators.Spin12(sym='dense')
    psi = peps.product_peps(geometry=geometry, vectors=opt.I())
    env = peps.EnvNTU(psi, which='NN')

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


    env = peps.EnvNTU(ancilla_psi, which='NN')
    opts_svd = {'D_total': 5}


    my_evolution_step(env, gates=gates, opts_svd=opts_svd)


    #print(env.psi[(0,0)].get_shape())
    # try to decompose the last leg
    #print(env.psi[(0,0)].unfuse_legs(axes=-1)) # the phsical has -1: outgoing, which make sense i guess, ancilla is 1 <- ingoing

    # check the first bond
    # bond_lr = peps._geometry.Bond(peps._geometry.Site(0,0), peps._geometry.Site(0,1))

    # Q0d, R0d, Q1d, R1d = decompose_A_and_B(env, bond_lr)
    # print('Q0d:', Q0d.get_shape())
    # print('R0d:', R0d.get_shape())
    # print('Q1d:', Q1d.get_shape())
    # print('R1d:', R1d.get_shape())

if __name__ == '__main__':
    main()
