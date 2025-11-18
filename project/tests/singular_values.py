import yastn
import numpy as np

config = yastn.make_config(sym='dense')
leg = yastn.Leg(config, s=1, t=(), D=(5, ))
tensor = yastn.rand(config, legs=(leg, leg.conj()))

u, s, v = tensor.svd(sU=-1)
print(len(np.diag(s.to_numpy())))