import yastn
import yastn.tn.fpeps as peps



def generate_random_psi(dimensions:dict):
    '''
    Geneerate the psi yastn.Peps with Checkerboard pattern with two random tensors A and B.
    '''

    config = yastn.make_config(sym='dense')
    
    leg_t = yastn.Leg(config, s=-1, t=(), D=(dimensions['t'], ))
    leg_l = yastn.Leg(config, s=1, t=(),  D=(dimensions['l'], ))
    leg_b = yastn.Leg(config, s=1, t=(),  D=(dimensions['b'], ))
    leg_r = yastn.Leg(config, s=-1, t=(), D=(dimensions['r'], ))
    leg_s = yastn.Leg(config, s=1, t=(),  D=(dimensions['s'], ))
    leg_a = yastn.Leg(config, s=-1, t=(), D=(dimensions['a'], ))


    tensor_A = yastn.rand(config, legs=[leg_t, leg_l, leg_b, leg_r, leg_s, leg_a])
    tensor_B = yastn.rand(config, legs=[leg_t, leg_l, leg_b, leg_r, leg_s, leg_a])

    geometry = peps.CheckerboardLattice()

    psi = peps.Peps(geometry=geometry, tensors={
        (0,0): tensor_A,
        (0,1): tensor_B
    })

    return psi