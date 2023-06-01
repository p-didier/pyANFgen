
import numpy as np
import soundfile as sf
import scipy.signal as sig
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class ANFgenConfig:
    """Class to store configuration for ANFgen"""
    # Sampling frequency [Hz]
    fs: int = 16000
    # Speed of sound [m/s]
    c: float = 343.0
    # DFT size
    N: int = 1024
    # Number of sensors
    M: int = 4
    # Inter-sensor distance [m]
    d: float = 0.1
    # Type of noise field ('spherical' or 'cylindrical')
    nfType: str = 'spherical'
    # Signal type ('noise' or 'babble')
    sigType: str = 'noise'
    # Babble file location
    babbleFile: str = ''
    # Fixed noise file location
    fixedNoiseFile: str = ''
    # Signal duration [s]
    T: float = 1.0
    # Random seed
    seed: int = None

    def __post_init__(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        self.L = int(self.T * self.fs)


def pyanfgen(cfg: ANFgenConfig, plot=False):

    # Generate base signals
    baseSigs = gen_base_sigs(cfg)

    # Generate matrix with desired spatial coherence
    dc = gen_coherence_matrix(cfg)

    # Generate sensor signals with desired spatial coherence
    x = mix_signals(baseSigs, dc, 'cholesky')

    # Compare desired and generated coherence
    if plot:
        compare(x, cfg)

    return x


def gen_base_sigs(cfg: ANFgenConfig):
    """Generate base signals for ANFgen."""
    if cfg.sigType == 'noise':
        if cfg.fixedNoiseFile == '':
            baseSigs = np.random.randn(cfg.L, cfg.M)
        else:
            baseSigs, fsData = sf.read(cfg.fixedNoiseFile, dtype='float32')
            if fsData != cfg.fs:
                raise ValueError(f'Sampling frequency of babble file ({fsData} Hz) does not match specified sampling frequency ({cfg.fs} Hz).')
    elif cfg.sigType == 'babble':
        if cfg.babbleFile == '':
            raise ValueError('Babble file location not specified.')
        else:
            fullBabble, fsData = sf.read(cfg.babbleFile, dtype='float32')
            if fsData != cfg.fs:
                raise ValueError(f'Sampling frequency of babble file ({fsData} Hz) does not match specified sampling frequency ({cfg.fs} Hz).')
            # Make zero-mean
            fullBabble = fullBabble - np.mean(fullBabble, axis=0)
            baseSigs = np.zeros((cfg.L, cfg.M))
            for m in range(cfg.M):
                baseSigs[:, m] = fullBabble[(m * cfg.L):((m + 1) * cfg.L)]
    return baseSigs


def gen_coherence_matrix(cfg: ANFgenConfig):
    """Generate matrix with desired spatial coherence."""
    ww = 2 * np.pi * cfg.fs * np.arange(cfg.N // 2 + 1) / cfg.N
    dc = np.zeros((cfg.M, cfg.M, cfg.N // 2 + 1))
    for p in range(cfg.M):
        for q in range(cfg.M):
            if p == q:
                dc[p, q, :] = np.ones((1, cfg.N // 2 + 1))
            else:
                if cfg.nfType == 'spherical':
                    dc[p, q, :] = np.sinc(ww * np.abs(p - q) * cfg.d / (cfg.c * np.pi))
                elif cfg.nfType == 'cylindrical':
                    raise NotImplementedError('Cylindrical noise field not implemented yet -- missing `bessel` function (see original MATLAB implementation of ANF-Generator).')
                else:
                    raise ValueError('Unknown noise field.')
    return dc


def mix_signals(n: np.ndarray, dc: np.ndarray, method='cholesky'):
    """Mix signals with desired spatial coherence."""
    nSensors = n.shape[1] # Number of sensors
    nFreqBins = (dc.shape[2] - 1) * 2 # Number of frequency bins
    originalLength = n.shape[0] # Original length of input signal
    # Compute short-time Fourier transform (STFT) of all input signals
    n = np.concatenate((np.zeros((nFreqBins // 2, nSensors)), n, np.zeros((nFreqBins // 2, nSensors))), axis=0)

    # MATLAB's `hanning` window -- https://stackoverflow.com/a/56485857 (accessed 2023-06-01)
    win = .5 * (1 - np.cos(2 * np.pi * np.arange(1, int(nFreqBins / 2 + 1)).T / (nFreqBins + 1)))
    win = np.concatenate((win, win[::-1]))

    _, _, nSTFT = sig.stft(
        n,
        window=win,
        nperseg=nFreqBins,
        noverlap=0.75 * nFreqBins,
        axis=0,
        boundary=None,
        padded=False,
        return_onesided=False
    )
    # nSTFT *= np.sum(win)  # Scale to match MATLAB's implementation
    # Rearrange dimensions of STFT matrix
    nSTFT = np.moveaxis(nSTFT, 1, -1)
    # Generate output signal in the STFT domain for each frequency bin k
    # mixedSTFT = np.zeros_like(nSTFT) # STFT output matrix
    mixedSTFT = np.zeros(
        (nFreqBins // 2 + 1, nSTFT.shape[1], nSTFT.shape[2]),
        dtype=complex
    )
    # for k in range(1, nFreqBins // 2 + 1):
    for k in range(1, mixedSTFT.shape[0]):
        if method == 'cholesky':
            Cmat = np.linalg.cholesky(dc[:, :, k])
            # Make upper triangular
            Cmat = Cmat.T
        elif method == 'eigen':
            w, v = np.linalg.eig(dc[:, :, k])
            Cmat = np.sqrt(w) * v.T
        else:
            raise ValueError('Unknown method specified.')
        mixedSTFT[k, :, :] = np.matmul(nSTFT[k, :, :], np.conj(Cmat))
    # mixedSTFT[(nFreqBins // 2 + 1):, :, :] = np.conj(
    #     mixedSTFT[(nFreqBins // 2 + 1):, :, :][::-1, :, :]
    # )
    # Compute inverse STFT
    _, x = sig.istft(
        mixedSTFT,
        window=win,
        nperseg=nFreqBins,
        noverlap=0.75 * nFreqBins,
        input_onesided=True,
        boundary=None,
        time_axis=1,
        freq_axis=0,
    )
    x = x[nFreqBins // 2:, :]
    x = x[:originalLength, :]
    # x = x[nFreqBins // 2:(-nFreqBins // 2), :]
    return x


def compare(x, cfg: ANFgenConfig):
    """Compare desired and generated coherence."""
    ww = 2 * np.pi * cfg.fs * np.arange(cfg.N // 2 + 1) / cfg.N
    sc_theory = np.zeros((cfg.M - 1, cfg.N // 2 + 1))
    sc_generated = np.zeros((cfg.M - 1, cfg.N // 2 + 1))

    # Calculate STFT and PSD of all output signals
    _, _, xSTFT = sig.stft(
        x,
        window=np.hanning(cfg.N),
        nperseg=cfg.N,
        noverlap=0.75 * cfg.N,
        padded=False,
        boundary=None,
        return_onesided=False,
        axis=0
    )
    # Rearrange dimensions of STFT matrix
    xSTFT = np.moveaxis(xSTFT, 1, -1)
    xSTFT = xSTFT[:cfg.N // 2 + 1, :, :]
    phi_x = np.mean(np.abs(xSTFT) ** 2, axis=1)

    # Calculate spatial coherence of desired and generated signals
    for m in range(cfg.M - 1):
        if cfg.nfType == 'spherical':
            sc_theory[m, :] = np.sinc(ww * (m + 1) * cfg.d / (cfg.c * np.pi))
        elif cfg.nfType == 'cylindrical':
            raise NotImplementedError('Cylindrical noise field not implemented yet -- missing `bessel` function (see original MATLAB implementation of ANF-Generator).')
        else:
            raise ValueError('Unknown noise field.')
        # Compute cross-PSD of x_1 and x_(m+1)
        psi_x = np.mean(xSTFT[:, :, 0] * np.conj(xSTFT[:, :, m + 1]), axis=1)
        # Compute real-part of complex coherence between x_1 and x_(m+1)
        sc_generated[m, :] = np.real(
            psi_x / np.sqrt(phi_x[:, 0] * phi_x[:, m + 1])
        ).T

    # Calculate normalized mean square error
    nmse = np.zeros((cfg.M, 1))
    for m in range(cfg.M - 1):
        nmse[m] = 10 * np.log10(
            np.sum(((sc_theory[m, :]) - (sc_generated[m, :])) ** 2) /\
                np.sum((sc_theory[m, :]) ** 2)
        )

    # Plot spatial coherence of two sensor pairs
    mm = min(2, cfg.M - 1)
    f = np.arange(0, stop=cfg.fs / 2 + 1, step=(cfg.fs / 2) / (cfg.N / 2))
    fig, axes = plt.subplots(mm, 1)
    fig.set_size_inches(6.5, 5.5)
    for m in range(mm):
        if mm == 1:
            currAx = axes
        else:
            currAx = axes[m]
        currAx.plot(f / 1000, sc_theory[m, :], '-k', linewidth=1.5)
        currAx.plot(f / 1000, sc_generated[m, :], '-.b', linewidth=1.5)
        currAx.set_xlabel('Frequency [kHz]')
        currAx.set_ylabel('Real(Spatial Coherence)')
        currAx.set_title(f'Inter sensor distance {(m + 1) * cfg.d} m')
        currAx.legend('Theory', f'Proposed Method (NMSE = {nmse[m]} dB)')
        currAx.grid(True)
    fig.tight_layout()
    plt.show()