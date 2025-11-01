import yastn

def apply_swap_gate(tensor: yastn.Tensor) -> yastn.Tensor:
    '''Apply the swap gate on the matrix tensor, see what happens'''
    return tensor.swap_gate(axes=(0,1))

def main():

    # creation of 2x2 tensor
    my_config = yastn.make_config(sym='dense')
    leg = yastn.Leg(my_config, t=(), D=(2,))
    tensor = yastn.rand(config=my_config, legs=[leg, leg.conj()])
    print(tensor.to_dense())
    print('---')
    print(apply_swap_gate(tensor).to_dense())

    # so it does not work as transpostion

if __name__ == '__main__':
    main()