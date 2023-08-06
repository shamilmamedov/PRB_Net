import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

from FEIN.utils.data import load_trajs
import data_preprocessing as data_pp


def main(n_trajs):
    trajs = load_trajs(n_trajs)
    # traj.plot('p_e')

    dt = 0.004
    yf, xf = [], []
    for traj in trajs:
        pe = np.asarray(traj.p_e) # p_e, dp_e, ddp_b
        N = pe.shape[0]

        yf.append(fft(pe, axis=0))
        xf.append(fftfreq(N, dt)[:N//2])

    fig, axs = plt.subplots(3,1, sharex=True)
    axs.reshape(-1,1)
    for k, ax in enumerate(axs):
        for n_traj, x, y in zip(n_trajs, xf, yf):
            N = y.shape[0]
            ax.plot(x, 2.0/N * np.abs(y[0:N//2,k]), label=f'traj {n_traj}')
        ax.legend(ncol=2)
        ax.set_xlim([0, 4])
        ax.set_ylim([0, 0.5])
        ax.grid()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    n_trajs = [2,4,6,14,16,19]
    main(n_trajs)