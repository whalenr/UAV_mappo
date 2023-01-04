import numpy as np
from environment2.Constant import N_user, N_ETUAV

horizontal_ue_loc = (np.random.rand(N_user, 2) - 0.5) * 2
horizontal_et_loc = (np.random.rand(N_ETUAV, 2) - 0.5) * 2

np.savetxt('horizontal_ue_loc.txt', horizontal_ue_loc)
np.savetxt('horizontal_et_loc.txt', horizontal_et_loc)
