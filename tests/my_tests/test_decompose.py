import yastn
from yastn.tn.fpeps._geometry import Site, Bond
from yastn.tn.fpeps._my_evolution import decompose_A_and_B
import yastn.tn.fpeps as peps
from tests.my_tests.helpers import generate_random_psi


def test_decompose_A_and_B():

    # Create some unique dimensions:
    dimensions = {
        't': 3,
        'l': 4,
        'b': 5,
        'r': 6,
        's': 7,
        'a': 8
    }

    psi = generate_random_psi(dimensions=dimensions)
    env = peps.EnvNTU(psi, which='NN')

    # 1. Check left-right dimension (horizontal)
    bond = Bond(Site(0,0), Site(0,1))
    Q0d, R0d, Q1d, R1d = decompose_A_and_B(env, bond=bond)

    # a) check the shapes
    assert Q0d.get_shape() == (3,4,5,6*8,7)
    assert R0d.get_shape() == (6*8,6,8)

    assert Q1d.get_shape() == (3,4*8,5,6,7)
    assert R1d.get_shape() == (4,4*8,8)

    # b) check signatures
    assert Q0d.s == (-1,1,1,-1,1)
    assert R0d.s == (1,-1,-1)

    assert Q1d.s == (-1,1,1,-1,1)
    assert R1d.s == (1,-1,-1)

    # 2. Check top-bottom dimension (horizontal)
    bond = Bond(Site(0,0), Site(1,0))
    Q0d, R0d, Q1d, R1d = decompose_A_and_B(env, bond=bond)

    # a) check the shapes
    assert Q0d.get_shape() == (3,4,5*8,6,7)
    assert R0d.get_shape() == (5*8,5,8)

    assert Q1d.get_shape() == (3*8,4,5,6,7)
    assert R1d.get_shape() == (3,3*8,8)

    # b) check signatures
    assert Q0d.s == (-1,1,1,-1,1)
    assert R0d.s == (-1,1,-1)

    assert Q1d.s == (-1,1,1,-1,1)
    assert R1d.s == (-1,1,-1)