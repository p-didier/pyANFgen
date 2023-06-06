# Purpose of script:
# Test pyANFgen.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import sys
import numpy as np
from pyanfgen.utils import ANFgenConfig, pyanfgen

M = 3  # number of sensors in network
SEED = 12345
MAX_DISTANCE = 2    # [m] max. inter-sensor distance
MIN_DISTANCE = 0.2  # [m] min. inter-sensor distance

def main():
    """Main function (called by default when running script)."""

    np.random.seed(SEED)

    # Create random distance matrix
    b = (MAX_DISTANCE - MIN_DISTANCE) * np.random.random(size=(M, M)) + MIN_DISTANCE
    d = (b + b.T) / 2  # make symmetric
    np.fill_diagonal(d, 0.)  # ensure that diagonal contains zeros

    # d = np.full(shape=(M, M), fill_value=0.4)
    # np.fill_diagonal(d, 0.)

    cfg = ANFgenConfig(
        fs=8000,
        c=340.0,
        N=256,
        M=M,
        d=d,
        nfType='spherical',
        sigType='noise',
        T=10.0,
        seed=0,
        # fixedNoiseFile='C:/Users/pdidier/Dropbox/_BELGIUM/KUL/SOUNDS_PhD/02_research/03_simulations/01_matlab/01_algorithms/99_third_parties/habets/ANF-Generator/baseSigFixedNoise.wav'
    )

    pyanfgen(cfg, plot=True)


if __name__ == '__main__':
    sys.exit(main())