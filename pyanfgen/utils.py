
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
    # Inter-sensor distance matrix [m]
    d: np.ndarray = np.array([0.])
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
        if isinstance(self.d, float) or isinstance(self.d, int):
            self.d = np.full(shape=(self.M, self.M), fill_value=self.d)
            np.fill_diagonal(self.d, 0.)
        if not np.allclose(self.d, self.d.T):
            raise ValueError('The distance matrix is not symmetric.')
        if self.d.shape[0] != self.M or self.d.shape[1] != self.M:
            raise ValueError(f'The provided distance matrix dimensions ({self.d.shape[0]}x{self.d.shape[1]}) do not match the number of mics ({self.M}).')
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
                    dc[p, q, :] = np.sinc(
                        # ww * np.abs(p - q) * cfg.d[p, q] / (cfg.c * np.pi)
                        ww * cfg.d[p, q] / (cfg.c * np.pi)
                    )
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
    n = np.concatenate((
        np.zeros((nFreqBins // 2, nSensors)),
        n,
        np.zeros((nFreqBins // 2, nSensors))
    ), axis=0)

    # MATLAB's `hanning` window
    # -- https://stackoverflow.com/a/56485857 (accessed 2023-06-01)
    win = .5 * (1 - np.cos(
        2 * np.pi * np.arange(1, int(nFreqBins / 2 + 1)).T / (nFreqBins + 1)
    ))
    win = np.concatenate((win, win[::-1]))

    _, _, nSTFT = sig.stft(
        n,
        window=win,
        nperseg=nFreqBins,
        noverlap=0.5 * nFreqBins,
        axis=0,
        boundary=None,
        padded=False,
        return_onesided=False
    )
    # Rearrange dimensions of STFT matrix
    nSTFT = np.moveaxis(nSTFT, 1, -1)
    # Generate output signal in the STFT domain for each frequency bin k
    mixedSTFT = np.zeros(
        (nFreqBins // 2 + 1, nSTFT.shape[1], nSTFT.shape[2]),
        dtype=complex
    )
    for k in range(1, mixedSTFT.shape[0]):
        if method == 'cholesky':
            Cmat = np.linalg.cholesky(
                nearestPD(dc[:, :, k])  # requires positive definite matrix `dc[:, :, k]`
            )
            # Make upper triangular
            Cmat = Cmat.T
        elif method == 'eigen':
            w, v = np.linalg.eig(dc[:, :, k])
            Cmat = np.sqrt(w) * v.T
        else:
            raise ValueError('Unknown method specified.')
        mixedSTFT[k, :, :] = np.matmul(nSTFT[k, :, :], np.conj(Cmat))
    # Compute inverse STFT
    _, x = sig.istft(
        mixedSTFT,
        window=win,
        nperseg=nFreqBins,
        noverlap=0.5 * nFreqBins,
        input_onesided=True,
        boundary=None,
        time_axis=1,
        freq_axis=0,
    )
    x = x[nFreqBins // 2:, :]
    x = x[:originalLength, :]
    return x


def compare(x, cfg: ANFgenConfig):
    """Compare desired and generated coherence."""
    ww = 2 * np.pi * cfg.fs * np.arange(cfg.N // 2 + 1) / cfg.N

    # Calculate STFT and PSD of all output signals
    _, _, xSTFT = sig.stft(
        x,
        window=np.hanning(cfg.N),
        nperseg=cfg.N,
        noverlap=0.5 * cfg.N,
        padded=False,
        boundary=None,
        return_onesided=False,
        axis=0
    )
    # Rearrange dimensions of STFT matrix
    xSTFT = np.moveaxis(xSTFT, 1, -1)
    xSTFT = xSTFT[:cfg.N // 2 + 1, :, :]
    phi_x = np.mean(np.abs(xSTFT) ** 2, axis=1)
    
    sc_theory = np.zeros((cfg.M, cfg.M, cfg.N // 2 + 1))
    sc_generated = np.zeros((cfg.M, cfg.M, cfg.N // 2 + 1))
    for p in range(cfg.M):
        for q in range(cfg.M):
            if p == q:
                sc_theory[p, q, :] = np.ones((1, cfg.N // 2 + 1))
                sc_generated[p, q, :] = np.ones((1, cfg.N // 2 + 1))
            else:
                if cfg.nfType == 'spherical':
                    # sc_theory[p, q, :] = np.sinc(ww * np.abs(p - q) * cfg.d[p, q] / (cfg.c * np.pi))
                    sc_theory[p, q, :] = np.sinc(ww * cfg.d[p, q] / (cfg.c * np.pi))
                elif cfg.nfType == 'cylindrical':
                    raise NotImplementedError('Cylindrical noise field not implemented yet -- missing `bessel` function (see original MATLAB implementation of ANF-Generator).')
                else:
                    raise ValueError('Unknown noise field.')
                # Compute cross-PSD of x_p and x_q
                psi_x = np.mean(xSTFT[:, :, p] * np.conj(xSTFT[:, :, q]), axis=1)
                # Compute real-part of complex coherence between x_1 and x_(m+1)
                sc_generated[p, q, :] = np.real(
                    psi_x / np.sqrt(phi_x[:, p] * phi_x[:, q])
                ).T

    nmse = np.zeros((cfg.M, cfg.M))
    for p in range(cfg.M):
        for q in range(cfg.M):
            if p == q:
                nmse[p, q] = 1
            else:
                nmse[p, q] = 10 * np.log10(
                    np.sum(((sc_theory[p, q, :]) - (sc_generated[p, q, :])) ** 2) /\
                        np.sum((sc_theory[p, q, :]) ** 2)
                )

    # Plot spatial coherence of every two sensor pairs
    f = np.arange(0, stop=cfg.fs / 2 + 1, step=(cfg.fs / 2) / (cfg.N / 2))
    fig, axes = plt.subplots(cfg.M, cfg.M, sharex=True, sharey=True)
    fig.set_size_inches(6.5, 5.5)
    for p in range(cfg.M):
        for q in range(cfg.M):
            if p != q:
                currAx = axes[p, q]
                currAx.plot(
                    f / 1000,
                    sc_theory[p, q, :],
                    '-k',
                    linewidth=1.5
                )
                currAx.plot(
                    f / 1000,
                    sc_generated[p, q, :],
                    '-.b',
                    linewidth=1.5
                )
                if p == cfg.M - 1:
                    currAx.set_xlabel('Frequency [kHz]')
                currAx.set_ylabel(f'$\\gamma_{{{p + 1}{q + 1}}}(\\omega)$')
                currAx.set_title(
                    f'$d_{{{p + 1}{q + 1}}}$ = {np.round(cfg.d[p, q], 2)} m'
                )
                if p == q and p == cfg.M - 1:
                    currAx.legend([
                        'Theory',
                        f'Proposed Method (NMSE = {np.round(nmse[p, q], 2)} dB)'
                    ])
                currAx.grid(True)
    fig.tight_layout()
    plt.show()


# From https://stackoverflow.com/a/43244194 
def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """
    def isPD(B):
        """Returns true when input is positive-definite, via Cholesky"""
        try:
            _ = np.linalg.cholesky(B)
            return True
        except np.linalg.LinAlgError:
            return False

    Bmat = (A + A.T) / 2
    _, s, Vmat = np.linalg.svd(Bmat)

    Hmat = np.dot(Vmat.T, np.dot(np.diag(s), Vmat))

    A2mat = (Bmat + Hmat) / 2

    A3mat = (A2mat + A2mat.T) / 2

    if isPD(A3mat):
        return A3mat

    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    Imat = np.eye(A.shape[0])
    k = 1
    while not isPD(A3mat):
        mineig = np.min(np.real(np.linalg.eigvals(A3mat)))
        A3mat += Imat * (-mineig * k**2 + spacing)
        k += 1

    return A3mat