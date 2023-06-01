# Purpose of script:
# Test pyANFgen.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import sys
from pyanfgen.utils import ANFgenConfig, pyanfgen

def main():
    """Main function (called by default when running script)."""

    cfg = ANFgenConfig(
        fs=8000,
        c=340.0,
        N=256,
        M=8,
        d=0.2,
        nfType='spherical',
        sigType='noise',
        T=10.0,
        seed=0,
        # fixedNoiseFile='C:/Users/pdidier/Dropbox/_BELGIUM/KUL/SOUNDS_PhD/02_research/03_simulations/01_matlab/01_algorithms/99_third_parties/habets/ANF-Generator/baseSigFixedNoise.wav'
    )

    pyanfgen(cfg)


if __name__ == '__main__':
    sys.exit(main())