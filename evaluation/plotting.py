import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib

import FEIN.utils.nn as nn_utils


params = {  #'backend': 'ps',
    "text.latex.preamble": r"\usepackage{gensymb} \usepackage{amsmath}",
    "axes.labelsize": 12,  # fontsize for x and y labels (was 10)
    "axes.titlesize": 8,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "text.usetex": True,
    "font.family": "serif",
}
matplotlib.rcParams.update(params)



def load_models_predictions(dlo:str, rollout_length):
    dir = 'evaluation/data/'
    names = [
        f'{dir}{dlo}_resnet_LFK_predictions_{rollout_length}steps.csv',
        f'{dir}{dlo}_resnet_NN_predictions_{rollout_length}steps.csv',
        f'{dir}{dlo}_rnn_LFK_predictions_{rollout_length}steps.csv',
        f'{dir}{dlo}_rnn_NN_predictions_{rollout_length}steps.csv',
        f'{dir}{dlo}_rfem_predictions_{rollout_length}steps.csv',
    ]
    dfs = []
    for name in names:
        try:
            dfs.append(pd.read_csv(name))
        except FileNotFoundError:
            pass
    shortcuts = ['FEI-\nResNet', 'ResNet', 'FEI-\nRNN', 'RNN', 'RFEM']
    return dict(zip(shortcuts[:len(dfs)], dfs))


def compare_models(save: bool = False):
    def _compute_l2_norm_of_prediction_error(df):
        Y_true = df[['pe_x', 'pe_y', 'pe_z']].to_numpy()
        Y_pred = df[['hat_pe_x', 'hat_pe_y', 'hat_pe_z']].to_numpy()

        E = Y_true - Y_pred
        return nn_utils.l2_norm(E[None,:,:])
    
    # Load models predictions
    dlo1 = 'aluminium_rod'
    dlo2 = 'pool_noodle'
    rollout_length = 250
    ar_pred_dic = load_models_predictions(dlo1, rollout_length)
    pn_pred_dic = load_models_predictions(dlo2, rollout_length)

    # Compute prediction errors
    ar_l2_norms = {}
    for k, ar_df in ar_pred_dic.items():
        ar_l2_norms[k] = 100*_compute_l2_norm_of_prediction_error(ar_df)

    pn_l2_norms = {}
    for k, pn_df in pn_pred_dic.items():
        pn_l2_norms[k] = 100*_compute_l2_norm_of_prediction_error(pn_df)    
   
    y_max = 16
    sns.set_palette('pastel')
    fig, ax = plt.subplots(figsize=(5,2))
    ax.axvspan(4.5, 10, facecolor='lightgrey', alpha=0.5)
    sns.boxplot(
        data=list(ar_l2_norms.values()) + list(pn_l2_norms.values()), 
        showfliers=False,
        palette=sns.color_palette(n_colors=5))
   
    plt.xticks([k for k in range(10)], list(ar_l2_norms.keys())*2)
    for k, v in ar_l2_norms.items():
        plt.text(x=list(ar_l2_norms.keys()).index(k), y=np.mean(v), s=f'{np.mean(v):.2f}', fontdict={"fontsize": 10})
    for k, v in pn_l2_norms.items():
        plt.text(x=list(ar_l2_norms.keys()).index(k)+5, y=max(np.mean(v), y_max)-7, s=f'{np.mean(v):.2f}', fontdict={"fontsize": 10})
    plt.ylim([0, y_max])
    plt.xlim([-0.5, 10])
    plt.ylabel(r'$||p_\mathrm{e} - \hat p_\mathrm{e}||_2$ [cm]', fontdict={"fontsize": 12})
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()

    if save:
        fig.savefig(f'evaluation/figures/prediction_error_box_plot.svg', bbox_inches='tight')


def plot_long_term_prediction(save: bool = False):
    dlo1 =  'aluminium_rod'
    dlo2 = 'pool_noodle'
    rollout_length = 1250
    ar_pred_dic = load_models_predictions(dlo1, rollout_length)
    pn_pred_dic = load_models_predictions(dlo2, rollout_length)


    dlo_iidx = {'aluminium_rod': 13761, 'pool_noodle': 0}
    dlo_r = {'aluminium_rod': 2, 'pool_noodle': 1}

    fig, axs = plt.subplots(3, 2, sharex=True, figsize=(10.5,4))
    axs = axs.T.reshape(-1)
    ylabels = [r'$\hat p_{e,x}$ [m]', r'$\hat p_{e,y}$ [m]', r'$\hat p_{e,z}$ [m]']
    for dlo, axs, pred_dic in zip([dlo1, dlo2], [axs[:3], axs[3:]], [ar_pred_dic, pn_pred_dic]):
        i_idx = dlo_iidx[dlo]
        f_idx = pred_dic['RNN']['pe_x'].shape[0]
        rollouts = (np.arange(i_idx, f_idx, step=rollout_length+1)[:, None] + 
                    np.arange(rollout_length+1))
        r = dlo_r[dlo]
        idxs = rollouts[r]
        
        time = (idxs-idxs[0])*0.004
        for y_lbl, ax, pe_axis in zip(ylabels, axs, ['pe_x', 'pe_y', 'pe_z']):
            ax.plot(time, pred_dic['RNN'][pe_axis][idxs], 'k-', lw=2, label='Measured')
            for k, df in pred_dic.items():
                ax.plot(time, df['hat_' + pe_axis][idxs], lw=1, label=k)
            ax.set_ylabel(y_lbl)
            ax.set_xlim([0, 5])
            ax.grid(alpha=0.25)
            # ax.legend(ncol=5)
        axs[-1].set_xlabel('time [s]')
    # specify spacing between legend entries
    axs[0].legend(ncol=5, handlelength=1.0, columnspacing=0.35, handletextpad=0.5, loc='upper center')
    plt.tight_layout()
    plt.show()

    if save:
        fig.savefig(f'evaluation/figures/long_term_prediction.svg', format='svg', bbox_inches='tight')


if __name__ == '__main__':
    plot_long_term_prediction(save=True)
    # compare_models(save=False)