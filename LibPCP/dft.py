"""
Source: PCP Notebooks (https://www.audiolabs-erlangen.de/PCP)
Module: LibPCP.dft
Author: Meinard Mueller, International Audio Laboratories Erlangen
License: The MIT license, https://opensource.org/licenses/MIT
"""

import numpy as np
from matplotlib import pyplot as plt
import LibPCP.signal


def plot_signal_e_k(ax, x, k, show_e=True, show_opt=False):
    """Plot signal and k-th DFT sinusoid

    Notebook: PCP_dft.ipynb

    Args:
        ax: Figure axis
        x: Signal
        k: Index of DFT
        show_e: Shows cosine and sine
        show_opt: Shows cosine with optimal phase
    """
    N = len(x)
    time_index = np.arange(N)
    plt.plot(time_index, x, 'k', marker='.', markersize='10', linewidth=2.0, label='$x$')
    plt.xlabel('Time (samples)')
    e_k = np.exp(2 * np.pi * 1j * k * time_index / N)
    c_k = np.real(e_k)
    s_k = np.imag(e_k)
    X_k = np.vdot(e_k, x)

    plt.title(r'k = %d: Re($X(k)$) = %0.2f, Im($X(k)$) = %0.2f, $|X(k)|$=%0.2f' %
              (k, X_k.real, X_k.imag, np.abs(X_k)))
    if show_e is True:
        plt.plot(time_index, c_k, 'r', marker='.', markersize='5',
                 linewidth=1.0, linestyle=':', label='$\mathrm{Re}(\overline{\mathbf{u}}_k)$')
        plt.plot(time_index, s_k, 'b', marker='.', markersize='5',
                 linewidth=1.0, linestyle=':', label='$\mathrm{Im}(\overline{\mathbf{u}}_k)$')
    if show_opt is True:
        phase_k = - np.angle(X_k) / (2 * np.pi)
        cos_k_opt = np.cos(2 * np.pi * (k * time_index / N - phase_k))
        d_k = np.sum(x * cos_k_opt)
        plt.plot(time_index, cos_k_opt, 'g', marker='.', markersize='5',
                 linewidth=1.0, linestyle=':', label='$\cos_{k, opt}$')
    plt.grid()
    plt.legend(loc='lower right')


def generate_matrix_dft(N, K):
    """Generate a DFT (discete Fourier transfrom) matrix

    Notebook: PCP_dft.ipynb

    Args:
        N: Number of samples
        K: Number of frequency bins

    Returns:
        dft: The DFT matrix
    """
    dft = np.zeros((K, N), dtype=np.complex128)
    time_index = np.arange(N)
    for k in range(K):
        dft[k, :] = np.exp(-2j * np.pi * k * time_index / N)
    return dft


def dft(x):
    """Compute the discete Fourier transfrom (DFT)

    Notebook: PCP_dft.ipynb

    Args:
        x: Signal to be transformed

    Returns:
        X: Fourier transform of x
    """
    x = x.astype(np.complex128)
    N = len(x)
    dft_mat = generate_matrix_dft(N, N)
    return np.dot(dft_mat, x)


def fft(x):
    """Compute the fast Fourier transform (FFT)

    Notebook: PCP_dft.ipynb

    Args:
        x: Signal to be transformed

    Returns:
        X: Fourier transform of x
    """
    x = x.astype(np.complex128)
    N = len(x)
    log2N = np.log2(N)
    assert log2N == int(log2N), 'N must be a power of two!'
    X = np.zeros(N, dtype=np.complex128)

    if N == 1:
        return x
    else:
        this_range = np.arange(N)
        A = fft(x[this_range % 2 == 0])
        B = fft(x[this_range % 2 == 1])
        range_twiddle_k = np.arange(N // 2)
        sigma = np.exp(-2j * np.pi * range_twiddle_k / N)
        C = sigma * B
        X[:N//2] = A + C
        X[N//2:] = A - C
        return X


def plot_signal_dft(t, x, X, ax_sec=False, ax_Hz=False, freq_half=False, figsize=(10, 2)):
    """Plotting function for signals and its magnitude DFT

    Notebook: PCP_dft.ipynb

    Args:
        t: Time axis (given in seconds)
        x: Signal
        X: DFT
        ax_sec: Plots time axis in seconds
        ax_Hz: Plots frequency axis in Hertz
        freq_half: Plots only low half of frequency coefficients
        figsize: Size of figure
    """
    N = len(x)
    if freq_half is True:
        K = N // 2
        X = X[:K]
    else:
        K = N

    plt.figure(figsize=figsize)
    ax = plt.subplot(1, 2, 1)
    ax.set_title('$x$ with $N=%d$' % N)
    if ax_sec is True:
        ax.plot(t, x, 'k', marker='.', markersize='3', linewidth=0.5)
        ax.set_xlabel('Time (seconds)')
    else:
        ax.plot(x, 'k', marker='.', markersize='3', linewidth=0.5)
        ax.set_xlabel('Time (samples)')
    ax.grid()

    ax = plt.subplot(1, 2, 2)
    ax.set_title('$|X|$')
    if ax_Hz is True:
        Fs = 1 / (t[1] - t[0])
        ax_freq = Fs * np.arange(K) / N
        ax.plot(ax_freq, np.abs(X), 'k', marker='.', markersize='3', linewidth=0.5)
        ax.set_xlabel('Frequency (Hz)')

    else:
        ax.plot(np.abs(X), 'k', marker='.', markersize='3', linewidth=0.5)
        ax.set_xlabel('Frequency (index)')
    ax.grid()
    plt.tight_layout()
    plt.show()


def exercise_freq_index(show_result=True):
    """Exercise 1: Interpretation of Frequency Indices
       Notebook: PCP_dft.ipynb"""
    if show_result is False:
        return

    Fs = 64
    dur = 2
    x, t = LibPCP.signal.generate_example_signal(Fs=Fs, dur=dur)
    X = fft(x)

    print('=== Plot with axes given in indices (Fs=64, dur=2) ===', flush=True)
    plot_signal_dft(t, x, X)

    print('=== Plot with axes given in seconds and Hertz (Fs=64, dur=2) ===', flush=True)
    plot_signal_dft(t, x, X, ax_sec=True, ax_Hz=True, freq_half=True)

    Fs = 32
    dur = 2
    x, t = LibPCP.signal.generate_example_signal(Fs=Fs, dur=dur)
    X = fft(x)

    print('=== Plot with axes given in indices (Fs=32, dur=2) ===', flush=True)
    plot_signal_dft(t, x, X)

    print('=== Plot with axes given in seconds and Hertz (Fs=32, dur=2) ===', flush=True)
    plot_signal_dft(t, x, X, ax_sec=True, ax_Hz=True, freq_half=True)


def exercise_missing_time(show_result=True):
    """Exercise 2: Missing Time Localization
       Notebook: PCP_dft.ipynb"""
    if show_result is False:
        return

    N = 256
    T = 6
    omega1 = 1
    omega2 = 5
    amp1 = 1
    amp2 = 0.5

    t = np.linspace(0, T, N)
    t1 = t[:N//2]
    t2 = t[N//2:]

    x1 = amp1 * np.sin(2*np.pi*omega1*t) + amp2 * np.sin(2*np.pi*omega2*t)
    x2 = np.concatenate((amp1 * np.sin(2*np.pi*omega1*t1), amp2 * np.sin(2*np.pi*omega2*t2)))

    X1 = fft(x1)
    X2 = fft(x2)

    print('=== Plot with axes given in indices ===')
    plot_signal_dft(t, x1, X1)
    plot_signal_dft(t, x2, X2)
    plt.show()

    print('=== Plot with axes given in seconds and Hertz ===')
    plot_signal_dft(t, x1, X1, ax_sec=True, ax_Hz=True, freq_half=True)
    plot_signal_dft(t, x2, X2, ax_sec=True, ax_Hz=True, freq_half=True)
    plt.show()


def exercise_chirp(show_result=True):
    """Exercise 3: Chirp Signal
       Notebook: PCP_dft.ipynb"""
    if show_result is False:
        return

    def generate_chirp_linear(t0=0, t1=1, N=128):
        """Generation chirp with linear frequency increase

        Notebook: PCP_DFT.ipynb

        Args:
            t_0: Start time (seconds)
            t_1: End time (seconds)
            N: Number of samples

        Returns:
            x: Generated chirp signal
            t: Time axis (in seconds)
        """
        t = np.linspace(t0, t1, N)
        x = np.sin(np.pi * t ** 2)
        return x, t

    def generate_chirp_plot_signal_dft(t0, t1, N, figsize=(10, 2)):
        x, t = generate_chirp_linear(t0=t0, t1=t1, N=N)
        X = fft(x)
        # plot_signal_dft(t, x, X, figsize=figsize)
        plot_signal_dft(t, x, X, ax_sec=True, ax_Hz=True, freq_half=True)

    generate_chirp_plot_signal_dft(t0=0, t1=2, N=128)
    generate_chirp_plot_signal_dft(t0=0, t1=4, N=128)
    generate_chirp_plot_signal_dft(t0=4, t1=8, N=128)
    generate_chirp_plot_signal_dft(t0=4, t1=8, N=256)


def exercise_inverse(show_result=True):
    """Exercise 4: Inverse DFT
       Notebook: PCP_dft.ipynb"""
    if show_result is False:
        return

    def generate_matrix_dft_inv(N, K):
        """Generates an IDFT (inverse discrete Fourier transfrom) matrix

        Notebook: PCP_dft.ipynb

        Args:
            N: Number of samples
            K: Number of frequency bins

        Returns:
            dft: The DFT matrix
        """
        dft_inv = np.zeros((K, N), dtype=np.complex128)
        time_index = np.arange(N)
        for k in range(K):
            dft_inv[k, :] = np.exp(2j * np.pi * k * time_index / N) / N
        return dft_inv

    N = 32
    dft_mat = generate_matrix_dft(N, N)
    dft_inv_mat = generate_matrix_dft_inv(N, N)

    A = np.matmul(dft_mat, dft_inv_mat)
    B = np.matmul(dft_inv_mat, dft_mat)
    I = np.eye(N)
    print('Comparison between DFT * DFT_inv and I:', np.allclose(A, I))
    print('Comparison between DFT_inv * DFT and I:', np.allclose(B, I))

    dft_inv_mat_np = np.linalg.inv(dft_mat)
    print('Comparison between DFT_inv and DFT_inv_via_np:',
          np.allclose(dft_inv_mat, dft_inv_mat_np))

    def fft_inv(x):
        """Compute the fast inverse Fourier transform (FFT)

        Notebook: PCP_dft.ipynb

        Args:
            x: Signal to be transformed

        Returns:
            X: Fourier transform of x
        """
        x = x.astype(np.complex128)
        N = len(x)
        log2N = np.log2(N)
        assert log2N == int(log2N), 'N must be a power of two!'
        X = np.zeros(N, dtype=np.complex128)

        if N == 1:
            return x
        else:
            this_range = np.arange(N)
            A = fft_inv(x[this_range % 2 == 0])
            B = fft_inv(x[this_range % 2 == 1])
            range_twiddle_k = np.arange(N // 2)
            sigma = np.exp(2j * np.pi * range_twiddle_k / N)
            C = sigma * B
            X[:N//2] = A + C
            X[N//2:] = A - C
            return X / 2

    N = 16
    x = np.arange(N).astype('float')
    X = fft(x)
    y = fft_inv(X)
    print('Example signal x:', x)
    print('Signal y = fft_inv(fft(x)):', y, sep='\n')
    print('Comparison between x and y:', np.allclose(x, y))
