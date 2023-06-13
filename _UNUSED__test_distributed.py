# Purpose of script:
# Test pyANFgen in the presence of distributed arrays.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import sys
import numpy as np
import scipy.signal as sig
from pyanfgen.utils import ANFgenConfig, pyanfgen

N_ARRAYS = 2
N_MIC_PER_ARRAY = [2, 3]
SEED = 12345
INTERARRAY_DISTANCES = [1]  # [m]
INTRAARRAY_DISTANCES = [0.2, 0.2]  # [m] one per array

def main():
    """Main function (called by default when running script)."""
    kwargs = {
        'fs': 8000,
        'c': 340.0,
        'N': 256,
        'nfType': 'spherical',
        'sigType': 'noise',
        'T': 10.0,
        'seed': SEED,
    }
    xAll = np.zeros((int(kwargs['fs'] * kwargs['T']), 0))
    totalMics = np.sum(N_MIC_PER_ARRAY)
    d = np.zeros((totalMics, totalMics))  # inter-mic distances
    for ii in range(N_ARRAYS):
        cfg = ANFgenConfig(
            M=N_MIC_PER_ARRAY[ii],
            d=INTRAARRAY_DISTANCES[ii],
            **kwargs
        )
        x = pyanfgen(cfg)
        xAll = np.concatenate((xAll, x), axis=1)
        # Compute inter-mic distances
        for jj in range(N_MIC_PER_ARRAY[ii]):
            for kk in range(N_MIC_PER_ARRAY[ii]):
                micIdx = np.sum(N_MIC_PER_ARRAY[:ii]) + jj
                micIdx2 = np.sum(N_MIC_PER_ARRAY[:ii]) + kk
                # d[micIdx, micIdx2] = TODO:TODO:TODO:
        stop = 1

    # Compute the coherence between each sensor pair
    # (for each frequency bin)
    compute_coherence(
        x,
        M=int(np.sum(N_MIC_PER_ARRAY)),
        fs=kwargs['fs'],
        N=kwargs['N'],
        nfType=kwargs['nfType'],
        d=d,
        c=kwargs['c']
    )

    stop = 1


def compute_coherence(x, M, fs, N, nfType, d, c):
    """Compute coherence between each sensor pair (for each frequency bin)."""
        # raise ValueError('Issue: matrix-like distances `d` not yet supported [PD:2023.06.06]')  # FIXME: account for this
    ww = 2 * np.pi * fs * np.arange(N // 2 + 1) / N

    # Calculate STFT and PSD of all output signals
    _, _, xSTFT = sig.stft(
        x,
        window=np.hanning(N),
        nperseg=N,
        noverlap=0.75 * N,
        padded=False,
        boundary=None,
        return_onesided=False,
        axis=0
    )
    # Rearrange dimensions of STFT matrix
    xSTFT = np.moveaxis(xSTFT, 1, -1)
    xSTFT = xSTFT[:N // 2 + 1, :, :]
    phi_x = np.mean(np.abs(xSTFT) ** 2, axis=1)
    
    sc_theory = np.zeros((M, M, N // 2 + 1))
    sc_generated = np.zeros((M, M, N // 2 + 1))
    for p in range(M):
        for q in range(M):
            if p == q:
                sc_theory[p, q, :] = np.ones((1, N // 2 + 1))
                sc_generated[p, q, :] = np.ones((1, N // 2 + 1))
            else:
                if nfType == 'spherical':
                    sc_theory[p, q, :] = np.sinc(ww * np.abs(p - q) * d[p, q] / (c * np.pi))
                elif nfType == 'cylindrical':
                    raise NotImplementedError('Cylindrical noise field not implemented yet -- missing `bessel` function (see original MATLAB implementation of ANF-Generator).')
                else:
                    raise ValueError('Unknown noise field.')
                # Compute cross-PSD of x_p and x_q
                psi_x = np.mean(xSTFT[:, :, p] * np.conj(xSTFT[:, :, q]), axis=1)
                # Compute real-part of complex coherence between x_1 and x_(m+1)
                sc_generated[p, q, :] = np.real(
                    psi_x / np.sqrt(phi_x[:, p] * phi_x[:, q])
                ).T


if __name__ == '__main__':
    sys.exit(main())