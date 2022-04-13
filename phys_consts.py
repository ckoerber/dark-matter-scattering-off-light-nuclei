#!/usr/bin/python
import numpy as np


_constants = {
    "mpi": np.float128(139.57018),
    "mN": np.float128(939.5653),
    "EDeut": np.float128(2.225),
    "fpi": np.float128(92.4),
    "gA": np.float128(1.29),
    "hbarc": np.float128(197.3269718),
}


class PhyiscalConstant(object):
    """
    Physical constant class. Initialized with the value in MeV
    """

    # -------------------------------------------------------
    def __init__(self, value, MeV=True):
        val = np.float128(value)
        self._val = val
        if MeV:
            self.MeV = val
            self.fm = val / _constants["hbarc"]
        else:
            self.MeV = val * _constants["hbarc"]
            self.fm = val

    # -------------------------------------------------------
    def __repr__(self):
        return str(self._val)


for const, val in _constants.items():
    globals()[const] = PhyiscalConstant(val)
