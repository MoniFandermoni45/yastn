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


        if dirn == 'h' or  dirn == 'lr':  # Horizontal gate, "lr" ordered

            # perform QR:
            # we can immedietely specify the Q axis, we keep ancillas always at the end
            Q0d, R0d = tensor_A.qr(axes=((0, 1, 2, 4), (3, 5)), sQ=-1, Qaxis=3)  # t l b rr s @ rr r a
            Q1d, R1d = tensor_B.qr(axes=((0, 2, 3, 4), (1, 5)), sQ=1, Qaxis=1, Raxis=1)  # t ll b r s @ l ll a


        else: # dirn == 'v':  # Vertical gate, "tb" ordered

            Q0d, R0d = tensor_A.qr(axes=((0, 1, 3, 4), (2, 5)), sQ=1, Qaxis=2)  # t l bb r s @ bb b a
            Q1d, R1d = tensor_B.qr(axes=((1, 2, 3, 4), (0, 5)), sQ=-1, Qaxis=0, Raxis=1)  # tt l b r s @ t tt a
    
    return Q0d, R0d, Q1d, R1d

def contract_back_reduced_tensors(Q0d: yastn.Tensor, Q1d: yastn.Tensor, r0d: yastn.Tensor, r1d: yastn.Tensor, dirn):
    '''
    Contracts back to obtain (tensor_A, tensor_B)
    They have unfused physical and ancilla indices, so both: t l b r s a
    '''

    if (dirn == 'lr' or dirn == 'h'): # left right bond / horizontal bond
        #Q0d: t l b rr s r0d: rr r a
        #Q1d: t ll b r s   r1d: l ll a
        tensor_A = yastn.tensordot(Q0d, r0d, axes=(3, 0)) # t l b s r a  # we use this notation
        tensor_A = tensor_A.transpose(axes=(0,1,2,4,3,5)) # t l b r s a

        tensor_B = yastn.tensordot(Q1d, r1d, axes=(1, 1)) # t b r s l a
        tensor_B = tensor_B.transpose(axes=(0,4,1,2,3,5)) # t l b r s a

    if (dirn == 'tb' or dirn == 'v'): # top bottom bond / vertical bond
        #Q0d: t l bb r s r0d: bb b a
        #Q1d: tt l b r s   r1d: t tt a
        tensor_A = yastn.tensordot(Q0d, r0d, axes=(2,0)) # t l r s b a
        tensor_A = tensor_A.transpose(axes=(0,1,4,2,3,5)) # t l b r s a

        tensor_B = yastn.tensordot(Q1d, r1d, axes=(0, 1)) # l b r s t a
        tensor_B = tensor_B.transpose(axes=(4,0,1,2,3,5)) # t l b r s a
    
    return tensor_A, tensor_B



# 2. Find the metric of the 

# TODO: change the env type,
def my_evolution_step(env: peps.EnvNTU, gates, opts_svd, methodType, method='mpo', fix_metric=0,
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
            
            # the unfuse of physical and ancilla happens here
            Q0d, R0d, Q1d, R1d = decompose_A_and_B(env, bond)

            # 2. Get the metric based on the sites s0, s1
            dirn = psi.nn_bond_dirn(bond) # take the direction of the bond

            if methodType == 'method_2':
                fgf = env.bond_metric(Q0=Q0d, Q1=Q1d, s0=s0, s1=s1, dirn=dirn)
            else:
                fgf = None
            # the metric for lr: [rr' ll'] [rr ll]
            # the metric for tb: [bb' tt'] [bb tt]

            # 3. Apply the metric to R0dR1d to get R0dR1d_tilde

            # 4. Put this to modified Yintai's procedure -> we get (pre)disentangler
            R0d, R1d, _, _ = my_apply_predisentangler(R0d, R1d, D_total, methodType=methodType, metric=fgf, pinv_cutoffs=pinv_cutoffs, dirn=dirn)
            tensor_A, tensor_B = contract_back_reduced_tensors(Q0d, Q1d, R0d, R1d, dirn)

            tensor_A = tensor_A.fuse_legs(axes=(0,1,2,3,(4,5)))
            tensor_B = tensor_B.fuse_legs(axes=(0,1,2,3,(4,5)))
            # Update the env tensors
            psi[s0] = tensor_A
            psi[s1] = tensor_B

            # 5. One way to continue is to take simply the svd of g applied to RR
            # 6. Other way is to apply it nevertheless, in such way update the tensors and then put this to standard NTU procedure.

            # NTU step
            info = peps.truncate_(env, opts_svd, (s0, s1), fix_metric, pinv_cutoffs, max_iter, tol_iter, initialization)
            infos.append(info)
        
    return infos


def my_apply_predisentangler(r0d: yastn.Tensor, r1d: yastn.Tensor, D_total, methodType, metric, pinv_cutoffs, dirn, max_iter=400, tol=1e-7):
    '''
    Helper function, returns updated by application of optimal predisentangler reduced tensors, ready for back contraction
    returns (r0d, r1d, diff, num_of_iter)
    '''

    # Update the reduced tensors
    r0d, r1d, diff, num_of_iter = my_predisentangler_iter(r0d, r1d, D_total, max_iter, methodType=methodType, metric=metric, tol=tol, pinv_cutoffs=pinv_cutoffs, dirn=dirn)

    return r0d, r1d, diff, num_of_iter

def my_predisentangler_iter(r0d:yastn.Tensor, r1d:yastn.Tensor, D_total, max_iter, methodType, pinv_cutoffs, dirn, metric=None, tol=1e-7):
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

    # Here we may apply the metric to r0dr1d to include the metric in the optimization process
    if methodType == 'method_2':

        r0dr1d = apply_bipartite_metric(fgf=metric, r0dr1d=r0dr1d, pinv_cutoffs=pinv_cutoffs, dirn=dirn)

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
        diff = (r0dr1d.transpose(axes=(0,1,3,2)).fuse_legs(axes=((0, 1), (2, 3))) - r0dr1d_new.fuse_legs(axes=((0, 1), (2, 3)))).norm(p="fro") / r0dr1d.transpose(axes=(0,1,3,2)).fuse_legs(axes=((0, 1), (2,3))).norm(p="fro")

        r0dr1d = r0dr1d_new
        r0dr1d = r0dr1d.transpose(axes=(0,1,3,2)) # rr a ll a'
        # we may probably transpose back here to make sure we match the pattern rr a ll a'

        if diff < tol:
            # if the difference is sufficiently small


            # Yintai version: r0dr1d: xx s a s' a' yy
            # My version    : r0dr1d: rr a ll a'
            #u, s, v = yastn.svd_with_truncation(r0dr1d, axes=((0, 1, 2), (3, 4, 5)), sU=r0d.s[1], D_total=r0d.get_shape(axes=1))

            if methodType == 'method_2':
                # apply the found predisentangler on the original r1dr1d
                r0dr1d = yastn.tensordot(r0d, r1d, axes=(1,0)) # rr a ll a' (My version) # we construct it back
                r0dr1d = yastn.tensordot(r0dr1d, g, axes=((1, 3), (1, 3))) # rr ll a a' my version
                r0dr1d = r0dr1d.transpose(axes=(0,2,1,3)) # rr a ll a'


            u, s, v = yastn.svd_with_truncation(r0dr1d, axes=((0,1), (2,3)), sU=r0d.s[1], Uaxis = 1, D_total=r0d.get_shape(axes=1)) # u: r rr a, v: l ll a'

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

def apply_bipartite_metric(fgf, r0dr1d: yastn.Tensor, pinv_cutoffs, dirn):
    '''
    Apply the bipartite metric to r0dr1d tensor
    '''
    fgf = fgf.g # take the proper tensor
    G = fgf.unfuse_legs(axes=(0, 1)) # lr: rr' ll' rr ll          tb: bb' tt' bb tt

    # if the direction if tb change the order of the metric
    # and maybe for convienience put unprimed at the beginning -> they are applied to nonconjugated
    if (dirn == 'tb' or dirn == 'v'):
        G = G.transpose(axes=(3,2,1,0)) # tt bb tt' bb'
    else: #for lr case
        G = G.transpose(axes=(2,3,0,1)) # rr ll rr' ll'

    # rank-1 approximation
    #Gremove = G.remove_zero_blocks() # needed for sth?
    Gremove = G
    G0, S, G1 = yastn.linalg.svd_with_truncation(Gremove, axes=((0, 2), (1,3)), policy='lowrank', D_block=1, D_total=1) # split onto two parts
    #fid = (S.norm() / G.norm()).item()
    #eat_metric_error = (max(0., 1 - fid ** 2)) ** 0.5

    # Remove connecting legs
    G0 = G0.remove_leg(axis=2) # rr rr'  or tt tt'
    G1 = G1.remove_leg(axis=0) # ll ll'  or bb bb'


    # Make sure it is hermitian
    G0 = G0 / G0.trace().to_number()
    G1 = G1 / G1.trace().to_number()
    G0 = (G0 + G0.H) / 2
    G1 = (G1 + G1.H) / 2
    #
    # F0 = R0.H @ G0 @ R0
    # F1 = R1 @ G1 @ R1.H
    #

    # print('G0 signature:', G0.s)
    # print('G1 signature:', G1.s)
    if (dirn == 'tb' or dirn == 'v'):
        G0 = G0.flip_signature()
        G1 = G1.flip_signature()

    # Verify what we are doing actually min(pinv_cutoffs)
    S0, U0 = G0.eigh_with_truncation(axes=(0, 1), tol=min(pinv_cutoffs), sU=r0dr1d.s[0])
    S1, U1 = G1.eigh_with_truncation(axes=(0, 1), tol=min(pinv_cutoffs), sU=r0dr1d.s[2])

    S0 = S0.sqrt()
    U0 = S0.broadcast(U0, axes=1) 

    S1 = S1.sqrt()
    U1 = S1.broadcast(U1, axes=1)
    #

    # In my version: r0dr1d: lr : rr a ll a'   or    tt a bb a'

    # Apply first part: right or top

    # we should change the signature in the case tb
    r0dr1d = yastn.tensordot(r0dr1d, U0, axes=(0, 0)) # a ll a' rr  (or a bb a' tt)
    r0dr1d = yastn.tensordot(r0dr1d, U1, axes=(1, 0)) # a a' rr ll  (or a a' tt bb)

    # Transpose back to required form
    r0dr1d = r0dr1d.transpose(axes=(2,0,3,1)) # rr a ll a'

    #W0, W1 = symmetrized_svd(S0.sqrt() @ U0.H, U1 @ S1.sqrt(), opts_svd, normalize=False)
    #p0, p1 = R0 @ U0, U1.H @ R1
    return r0dr1d



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


    #my_evolution_step(env, gates=gates, opts_svd=opts_svd)


    #print(env.psi[(0,0)].get_shape())
    # try to decompose the last leg
    #print(env.psi[(0,0)].unfuse_legs(axes=-1)) # the phsical has -1: outgoing, which make sense i guess, ancilla is 1 <- ingoing

    # check the first bond
    bond_lr = peps._geometry.Bond(peps._geometry.Site(0,0), peps._geometry.Site(1,0))

    original_tensor_A = ancilla_psi[(0,0)]
    original_tensor_B = ancilla_psi[(0,1)]

    print('original_tensor_A:', original_tensor_A.get_shape())
    print('original_tensor_B:', original_tensor_B.get_shape())

    Q0d, R0d, Q1d, R1d = decompose_A_and_B(env, bond_lr)
    print('Q0d:', Q0d.get_shape())
    print('R0d:', R0d.get_shape())
    print('Q1d:', Q1d.get_shape())
    print('R1d:', R1d.get_shape())

    tensor_A, tensor_B = contract_back_reduced_tensors(Q0d, Q1d, R0d, R1d, dirn='tb')
    print('tensorA:', tensor_A.get_shape())
    print('tensorB:', tensor_B.get_shape())

    tensor_A = tensor_A.fuse_legs(axes=(0,1,2,3,(4,5)))
    tensor_B = tensor_B.fuse_legs(axes=(0,1,2,3,(4,5)))

    print((tensor_A - original_tensor_A).norm())
    print((tensor_B - original_tensor_B).norm())

if __name__ == '__main__':
    main()
