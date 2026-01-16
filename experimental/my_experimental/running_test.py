import numpy as np
import yastn
import yastn.tn.fpeps as fpeps
import logging

print('Yastn version:', yastn.version_tuple) # Testing if YASTN launches

D = 8
chi = 16
sym = "U1xU1xZ2"
coef = 0.25
J = 0.5
t = 1
beta_target = 0.05 # what final beta to reach # hence only two steps
dbeta = 0.025 # imaginary time step
ntu_environment = "NN+" # contraction scheme

tot_sites = int(2 * 3)
net = fpeps.SquareLattice((2,3), 'obc') # use the open boundary conditions # 2x3 lattice

# Import the operators:
#TODO: Add **config_kwargs when needed - rewrite as function
opt = yastn.operators.SpinfulFermions_tJ(sym=sym)


I = opt.I()
c_up, c_dn = opt.c(spin='u'), opt.c(spin='d') # annihilation operators for spin up and down
cdag_up, cdag_dn = opt.cp(spin='u'), opt.cp(spin='d') # creation operators for spin up and down

n_up, n_dn = opt.n(spin='u'), opt.n(spin='d') # particle number operators

n = n_up + n_dn # total particle number operator
Sz, Sp, Sm = opt.Sz(), opt.Sp(), opt.Sm()

psi = fpeps.product_peps(net, I) # intial product state # legs are fused

num_steps = round(beta_target / dbeta)
dbeta = beta_target / num_steps
print('Number of steps:', num_steps)


# Generation of gates:


mu = 1 # chemical potential
gates = [fpeps.gates.gate_nn_tJ(J, t, t, mu/4, mu/4, mu/4, mu/4, dbeta * coef, I, c_up, cdag_up, c_dn, cdag_dn, bond) for bond in net.bonds()]
# correct boundary terms with local chemical potential
gates += [fpeps.gates.gate_local_occupation(mu/2, dbeta * coef, I, n, site=(0, 0)),
            fpeps.gates.gate_local_occupation(mu/4, dbeta * coef, I, n, site=(0, 1)),
            fpeps.gates.gate_local_occupation(mu/2, dbeta * coef, I, n, site=(0, 2)),
            fpeps.gates.gate_local_occupation(mu/2, dbeta * coef, I, n, site=(1, 0)),
            fpeps.gates.gate_local_occupation(mu/4, dbeta * coef, I, n, site=(1, 1)),
            fpeps.gates.gate_local_occupation(mu/2, dbeta * coef, I, n, site=(1, 2))]


# symmetrize
gates = gates + gates[::-1] # 

env_evolution = fpeps.EnvNTU(psi, which=ntu_environment)

opts_svd = {"D_total": D, 'tol_block': 1e-15}

beta = 0 # initial beta
for _ in range(num_steps):
    beta += dbeta
    logging.info("beta = %0.3f" % beta)
    fpeps.evolution_step_(env_evolution, gates, opts_svd=opts_svd) # we need to call this function

# Calculate observables with ctm
max_sweeps = 10
tol_exp = 1e-8
opts_svd_ctm = {'D_total': chi} # maximal bond dimension

# Now we create another environment, this time for CTM
env = fpeps.EnvCTM(psi, init='eye')
env.update_()
