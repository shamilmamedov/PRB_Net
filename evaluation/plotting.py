import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib

import FEIN.utils.nn as nn_utils


params = {  #'backend': 'ps',
    "text.latex.preamble": r"\usepackage{gensymb} \usepackage{amsmath}",
    "axes.labelsize": 10,  # fontsize for x and y labels (was 10)
    "axes.titlesize": 8,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "text.usetex": True,
    "font.family": "sans serif"
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


def _compute_l2_norm_of_prediction_error(df, var='pe'):
    Y_true = df[[f'{var}_x', f'{var}_y', f'{var}_z']].to_numpy()
    Y_pred = df[[f'hat_{var}_x', f'hat_{var}_y', f'hat_{var}_z']].to_numpy()

    E = Y_true - Y_pred
    return nn_utils.l2_norm(E[None,:,:])


def compare_models(save: bool = False):   
    # Load models predictions
    dlo1 = 'aluminium_rod'
    dlo2 = 'pool_noodle'
    rollout_length = 251
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
    fig, ax = plt.subplots(figsize=(4.5,2))
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
    rollout_length = 1251
    ar_pred_dic = load_models_predictions(dlo1, rollout_length)
    pn_pred_dic = load_models_predictions(dlo2, rollout_length)


    dlo_iidx = {'aluminium_rod': 13761, 'pool_noodle': 0}
    dlo_r = {'aluminium_rod': 2, 'pool_noodle': 1}

    fig, axs = plt.subplots(3, 2, sharex=True, figsize=(7.5,3))
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
            line_meas,  = ax.plot(time, pred_dic['RNN'][pe_axis][idxs], 'k-', lw=2, label='Measured')
            line_models = []
            for k, df in pred_dic.items():
                line_k,  = ax.plot(time, df['hat_' + pe_axis][idxs], lw=1, label=k)
                line_models.append(line_k)
            if dlo == 'aluminium_rod':
                ax.set_ylabel(y_lbl, fontdict={"fontsize": 10})
            ax.set_xlim([0, 5])
            ax.grid(alpha=0.25)
            # ax.legend(ncol=5)
        axs[-1].set_xlabel('time [s]')
    # axs[0].legend(ncol=1, handlelength=1.0, columnspacing=0.35, handletextpad=0.5, loc='center right', bbox_to_anchor=(1.0, 0.5))
    lines = [line_meas] + line_models
    labels = [l.get_label() for l in lines]
    fig.legend(lines, labels, ncol=6, loc='upper center', bbox_to_anchor=(0.5, 1.0))
    plt.tight_layout()
    plt.show()

    if save:
        fig.savefig(f'evaluation/figures/long_term_prediction.svg', format='svg', bbox_inches='tight')


def accuracy_on_different_rollout_lengths(dlo: str = 'pool_noodle', var: str = 'dpe'):
    rollout_lengths = [251, 501, 1251, 2501, 5001]
    rollout_predictions = dict()
    for r in rollout_lengths:
        rollout_predictions[r] = load_models_predictions(dlo, r)

    # Compute prediction errors
    l2_norms = {}
    for r, pred_dic in rollout_predictions.items():
        l2_norms[r] = {}
        for k, df in pred_dic.items():
            l2_norms[r][k] = 100*_compute_l2_norm_of_prediction_error(df, var)

    # Create a dataframe with columns rollout_length and rows model. Each cell contains the mean prediction error
    results_df = pd.DataFrame(index=rollout_lengths, columns=list(rollout_predictions[251].keys()))
    for r, pred_dic in rollout_predictions.items():
        for k, df in pred_dic.items():
            mean_pe_str = np.array2string(np.mean(l2_norms[r][k]), precision=1)
            std_pe_str = np.array2string(np.std(l2_norms[r][k]), precision=1)
            results_df.loc[r, k] = (rf'{mean_pe_str}')# ± {std_pe_str}')
    results_df = results_df.rename(columns={'FEI-\nResNet': 'FEI-ResNet', 'FEI-\nRNN': 'FEI-RNN'})
    results_df = results_df.T
    print(results_df)


if __name__ == '__main__':
    # accuracy_on_different_rollout_lengths()
    plot_long_term_prediction(save=True)
    # compare_models(save=False)