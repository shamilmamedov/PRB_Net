import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib

import FEIN.utils.nn as nn_utils


params = {  #'backend': 'ps',
    "text.latex.preamble": r"\usepackage{gensymb} \usepackage{amsmath}",
    "axes.labelsize": 11,  # fontsize for x and y labels (was 10)
    "axes.titlesize": 8,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "text.usetex": True,
    "font.family": "Times New Roman",
    "font.size": 11,
}
matplotlib.rcParams.update(params)

COLORS = [
    (0.368, 0.507, 0.71), (0.881, 0.611, 0.142), (0.923, 0.386,0.209),
    (0.56, 0.692, 0.195),(0.528, 0.471, 0.701), (0.772, 0.432,0.102),
    (0.572, 0.586, 0.)
]


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
    shortcuts = ['PRBN-\nResNet', 'ResNet', 'PRBN-\nRNN', 'RNN', 'PRB']
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
   
    y_max = 15
    sns.set_palette('pastel')  # Blues pastel
    fig, ax = plt.subplots(figsize=(5, 1.8))
    ax.axvspan(4.5, 10, facecolor='lightgrey', alpha=0.5)
    sns.boxplot(
        data=list(ar_l2_norms.values()) + list(pn_l2_norms.values()), 
        showfliers=False,
        width=0.5,
        palette=sns.color_palette(n_colors=5))
    # In the boxplot set the width of the boxes
    # Put a text in the top in the first half of the plot
    ax.text(0.55, 13, r'$\mathrm{aluminium\ rod}$', fontsize=12)
    ax.text(5.5, 13, r'$\mathrm{foam\ cylinder}$', fontsize=12)

    plt.xticks([k for k in range(10)], list(ar_l2_norms.keys())*2, rotation=45, fontdict={"fontsize": 8})
    # for k, v in ar_l2_norms.items():
    #     plt.text(x=list(ar_l2_norms.keys()).index(k), y=np.median(v), s=f'{np.median(v):.2f}', fontdict={"fontsize": 10})
    # for k, v in pn_l2_norms.items():
    #     plt.text(x=list(ar_l2_norms.keys()).index(k)+5, y=np.median(v), s=f'{np.median(v):.2f}', fontdict={"fontsize": 10})
    plt.ylim([0, y_max])
    plt.xlim([-0.5, 10])
    plt.ylabel(r'$||p_\mathrm{e} - \hat p_\mathrm{e}||_2$ [cm]', fontdict={"fontsize": 12})
    plt.grid(alpha=0.25)
    plt.tight_layout(pad=0.2)
    plt.show()

    if save:
        fig.savefig(f'evaluation/figures/predictions_box_plot.pdf', bbox_inches='tight')


def plot_long_term_prediction(save: bool = False):
    dlo1 =  'aluminium_rod'
    dlo2 = 'pool_noodle'
    rollout_length = 1251
    ar_pred_dic = load_models_predictions(dlo1, rollout_length)
    pn_pred_dic = load_models_predictions(dlo2, rollout_length)


    dlo_iidx = {'aluminium_rod': 13761, 'pool_noodle': 0}
    dlo_r = {'aluminium_rod': 2, 'pool_noodle': 1}

    fig, axs = plt.subplots(3, 2, sharex=True, figsize=(5,2.5))
    axs = axs.T.reshape(-1)
    axs[3].set_yticks([1.2, 1.5, 1.8])
    axs[5].set_yticks([-0.3, -0.5])
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
            for i, (k, df) in enumerate(pred_dic.items()):
                line_k,  = ax.plot(time, df['hat_' + pe_axis][idxs], lw=1, color=COLORS[i], label=k)
                line_models.append(line_k)
            if dlo == 'aluminium_rod':
                ax.set_ylabel(y_lbl, fontdict={"fontsize": 11})
            ax.set_xlim([0, 5])
            ax.grid(alpha=0.25)
            # ax.legend(ncol=5)
        axs[-1].set_xlabel(r'$\mathrm{time}$ [s]', fontdict={"fontsize": 10})
    
    # axs[0].legend(ncol=1, handlelength=1.0, columnspacing=0.35, handletextpad=0.5, loc='center right', bbox_to_anchor=(1.0, 0.5))
    lines = [line_meas] + line_models
    labels = [l.get_label() for l in lines]
    fig.legend(lines, labels, ncol=6, columnspacing=0.5, borderpad=0.25, loc='upper center', bbox_to_anchor=(0.55, 1.12), fontsize=8)
    # fig.subplots_adjust(top=0.85)

    plt.tight_layout(pad=0.2)
    plt.show()

    if save:
        fig.savefig(f'evaluation/figures/long_term_predictions.pdf', format='pdf', bbox_inches='tight')


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


def plot_accuracy_vs_rollout_length(save_fig: bool = False):
    rollout_lengths = [251, 501, 1251, 2501, 5001]
    rollout_lenghts_in_sec = [int((r-1)*0.004) for r in rollout_lengths]
    ar_rollout_predictions = dict()
    pn_rollout_predictions = dict()
    for r, r_in_secs in zip(rollout_lengths, rollout_lenghts_in_sec):
        ar_rollout_predictions[r_in_secs] = load_models_predictions('aluminium_rod', r)
        pn_rollout_predictions[r_in_secs] = load_models_predictions('pool_noodle', r)

    ar_l2_norms = dict()
    for r, ar_df in ar_rollout_predictions.items():
        for model, df in ar_df.items():
            if model in ar_l2_norms:
                ar_l2_norms[model][r] = 100*np.mean(_compute_l2_norm_of_prediction_error(df))
            else:
                ar_l2_norms[model] = {r: 100*np.mean(_compute_l2_norm_of_prediction_error(df))}

    pn_l2_norms = dict()
    for r, pn_df in pn_rollout_predictions.items():
        for model, df in pn_df.items():
            if model in pn_l2_norms:
                pn_l2_norms[model][r] = 100*np.mean(_compute_l2_norm_of_prediction_error(df))
            else:
                pn_l2_norms[model] = {r: 100*np.mean(_compute_l2_norm_of_prediction_error(df))}

    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(5,1.8))
    for i, (model, l2_norms) in enumerate(ar_l2_norms.items()):
        axs[0].plot(list(l2_norms.keys()), list(l2_norms.values()), 'o-', color=COLORS[i], label=model)
    for i, (model, l2_norms) in enumerate(pn_l2_norms.items()):
        axs[1].plot(list(l2_norms.keys()), list(l2_norms.values()), 'o-', color=COLORS[i], label=model)
    for ax in axs:
        ax.set_xticks(rollout_lenghts_in_sec)
        ax.set_yscale('log')
        ax.grid(alpha=0.25)
        ax.set_xlabel('prediction horizon [s]')
        ax.set_xlim([1, 20])
        ax.set_ylim([2.5, 100])
    # place the legend outside on the right of the plot
    axs[1].legend(loc='upper left', bbox_to_anchor=(1, 1), labelspacing=1.0)
    axs[0].set_ylabel(r'$\|p_\mathrm{e} - \hat p_\mathrm{e}\|_2$ [cm]')
    axs[0].text(2, 67, r'$\mathrm{aluminium\ rod}$')
    axs[1].text(2, 67, r'$\mathrm{foam\ cylinder}$')
    plt.tight_layout(pad=0.2)
    plt.show()

    if save_fig:
        fig.savefig('evaluation/figures/long_term_preds.pdf', format='pdf', dpi=600, bbox_inches='tight')


def how_nseg_affects_predictions(save_fig: bool = False):
    def load_predictions(dlo, dyn, dec, n_seg):
        dir = 'evaluation/data/'
        name = f'{dir}{dlo}_{dyn}_{dec}_{n_seg}seg_predictions_251steps.csv'
        return pd.read_csv(name)

    dlo1 = 'aluminium_rod'
    dlo2 = 'pool_noodle'
    dyn = 'rnn'
    dec = 'LFK'
    n_segs = [2, 5, 7, 10, 20]

    # Load predictions
    ar_preds, pn_preds = dict(), dict()
    for n_seg in n_segs:
        ar_preds[n_seg] = load_predictions(dlo1, dyn, dec, n_seg)
        pn_preds[n_seg] = load_predictions(dlo2, dyn, dec, n_seg)
    
    # Compute prediction errors
    ar_pos_err_norms_in_cm = dict()
    ar_vel_err_norms_in_cms = dict()
    pn_pos_err_norms_in_cm = dict()
    pn_vel_err_norms_in_cms = dict()
    for n_seg in n_segs:
        ar_pos_err_norms_in_cm[n_seg] = 100.*np.mean(_compute_l2_norm_of_prediction_error(ar_preds[n_seg], 'pe'))
        ar_vel_err_norms_in_cms[n_seg] = 100.*np.mean(_compute_l2_norm_of_prediction_error(ar_preds[n_seg], 'dpe'))
        pn_pos_err_norms_in_cm[n_seg] = 100.*np.mean(_compute_l2_norm_of_prediction_error(pn_preds[n_seg], 'pe'))
        pn_vel_err_norms_in_cms[n_seg] = 100.*np.mean(_compute_l2_norm_of_prediction_error(pn_preds[n_seg], 'dpe'))

    pos_err = [ar_pos_err_norms_in_cm, pn_pos_err_norms_in_cm]
    vel_err = [ar_vel_err_norms_in_cms, pn_vel_err_norms_in_cms]

    # Plot mean l2 norm of prediction error
    fig, axs = plt.subplots(1,2,figsize=(5,1.8))
    axs.reshape(-1)
    for ax, pos_err, vel_err in zip(axs, pos_err, vel_err):
        ax.plot(pos_err.keys(), list(pos_err.values()), 'o-', color=COLORS[0])
        # add second y-axis
        ax2 = ax.twinx()
        ax2.plot(vel_err.keys(), list(vel_err.values()), 'o-', color=COLORS[3])
        ax.grid(alpha=0.25)
        ax.set_xlabel(r'$n_{\mathrm{el}}$', fontdict={"fontsize": 11})
        ax.tick_params(axis='y', labelcolor=COLORS[0])
        ax2.tick_params(axis='y', labelcolor=COLORS[3])
    ax2.set_ylabel(r'$\|\dot p_\mathrm{e} - \hat{\dot p}_\mathrm{e}\|_2$ [cm/s]', fontdict={'color': COLORS[3], "fontsize": 11})
    axs[0].set_ylabel(r'$\|p_\mathrm{e} - \hat p_\mathrm{e}\|_2$ [cm]', fontdict={'color': COLORS[0], "fontsize": 11})
    axs[0].text(0.55, 0.8, r'$\mathrm{aluminium\ rod}$', fontsize=12, ha='center', transform=axs[0].transAxes)
    axs[1].text(0.55, 0.8, r'$\mathrm{foam\ cylinder}$', fontsize=12, ha='center', transform=axs[1].transAxes)
    plt.tight_layout(pad=0.2)
    plt.show()

    if save_fig:
        fig.savefig('evaluation/figures/n_seg_vs_error.pdf', format='pdf', dpi=600, bbox_inches='tight')


if __name__ == '__main__':
    # accuracy_on_different_rollout_lengths()
    # plot_long_term_prediction(save=True)
    # compare_models(save=True)
    # how_nseg_affects_predictions(save_fig=True)
    plot_accuracy_vs_rollout_length(save_fig=True)