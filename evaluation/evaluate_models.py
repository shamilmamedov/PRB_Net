import yaml
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import pandas as pd
import time
import matplotlib
import pickle

from training import preprocess_data as data_pp
from training.run_experiment import get_model
import FEIN.utils.nn as nn_utils
import FEIN.utils.data as data_utils
from FEIN.rfem_kinematics import models
from FEIN.rfem_kinematics.visualization import visualize_robot


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


def compute_l2_norm_of_pos_prediction_error(
        Y_true: jnp.ndarray, 
        Y_preds: List[jnp.ndarray]
    ):
    out = []
    for Y_pred in Y_preds:
        out.append(
            nn_utils.l2_norm(Y_true[:,:,:3] - Y_pred[:,:,:3]).reshape(-1)
        )
    return out


def compute_l2_norm_of_vel_prediction_error(
        Y_true: jnp.ndarray, 
        Y_preds: List[jnp.ndarray]
    ):
    out = []
    for Y_pred in Y_preds:
        out.append(
            nn_utils.l2_norm(Y_true[:,:,3:] - Y_pred[:,:,3:])
        )
    return out


def get_model_configs(dlo: str, n_seg: int, dyn: str, dec: str) -> dict:
    # Construct config file name
    prfx = ''
    if dlo == 'pool_noodle': prfx = 'PN_'
    config_name = f'{dlo}/{prfx}{dyn}_{n_seg}seg_{dec}.yml'

    # Construct config file path
    configs_dir = 'training/experiment_configs/'
    experiment_config_path = configs_dir + config_name
    
    # Load config file
    with open(experiment_config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def load_trained_model(config: dict) -> eqx.Module:
    # Load mode skeletons
    model_pytree = get_model(config)

    # Deserialoze saved trained models
    models_dir = 'training/saved_models/'
    model_path = models_dir + config['name'] + '.eqx'
    trained_model = eqx.tree_deserialise_leaves(model_path, model_pytree)    

    return trained_model


def evaluate_models(models: List, data):
    X, Y_pred = [], []
    for m in models:
        X_, Y_ = jax.vmap(m)(data.U_encoder[:,0,:], data.U_dyn, data.U_decoder)
        X.append(X_)
        Y_pred.append(Y_)
    return X, Y_pred


def compute_performance_metrics(Y, Y_pred_models):
    rmse, mae = [], []
    for Y_pred in Y_pred_models:
        rmse.append(
            jnp.sqrt(nn_utils.mse_loss(Y, Y_pred)).item()
        )
        mae.append(
            nn_utils.mae_loss(Y, Y_pred).item()
        )
    return rmse, mae


def get_data_used_for_training(config: dict) -> data_utils.DLODataset:
    rollout_length = config['rollout_length']
    data_key = jax.random.PRNGKey(config['data_seed'])
    data_key, key = jax.random.split(data_key)

    if config['DLO'] == 'aluminium-rod':
        val_size = 0.15
        trajs = data_utils.load_trajs(config['train_trajs'], config['DLO'])
        train_trajs, val_trajs = data_pp.split_trajs_into_train_val(trajs, val_size, key)
    elif config['DLO'] == 'pool-noodle':
        train_trajs = data_utils.load_trajs(config['train_trajs'], config['DLO'])
        val_trajs = data_utils.load_trajs(config['val_trajs'], config['DLO'])
    else:
        raise ValueError('Please specify a valid DLO in the config file')

    train_data, val_data = data_pp.construct_train_val_datasets(
        train_trajs, val_trajs, rollout_length
    )
    return train_data


def get_test_data(config: dict, window: str, x_rollout: int) -> data_utils.DLODataset:
    # Load train and val data to get sacalars
    train_data = get_data_used_for_training(config)

    train_rollout_length = config['rollout_length']
    test_rollout_length = x_rollout * train_rollout_length
    
    n_test_trajs = config['test_trajs']
    test_trajs = data_utils.load_trajs(n_test_trajs, config['DLO'])
    
    test_data = data_pp.construct_test_dataset_from_trajs(
        test_trajs, test_rollout_length, train_data, window, scale_outputs=False
    )
    return test_data


def get_panda_rollouts(config: dict, window: str, x_rollout: int):
    window_divider = {
        'rolling':data_pp.divide_into_rolling_windows_and_stack,
        'sliding': data_pp.divide_into_sliding_windows_and_stack
    }

    train_rollout_length = config['rollout_length']
    test_rollout_length = x_rollout * train_rollout_length
    
    n_test_trajs = config['test_trajs']
    test_trajs = data_utils.load_trajs(n_test_trajs, config['DLO'])
    
    q_panda = [t.q_p for t in test_trajs]
    out = window_divider[window](q_panda, test_rollout_length)
    return out


def save_predictions_to_csv(dlo: str, model:str, Y_true, Y_pred, X_pred):
    rollout_length = Y_true.shape[1]
    nx = X_pred.shape[2]
    n_seg = nx // 4

    y_meas_cols = ['pe_x', 'pe_y', 'pe_z', 'dpe_x', 'dpe_y', 'dpe_z']
    y_pred_cols = ['hat_pe_x', 'hat_pe_y', 'hat_pe_z', 'hat_dpe_x', 'hat_dpe_y', 'hat_dpe_z']
    q_cols = [f'q_{i}' for i in range(2*n_seg)]
    dq_cols = [f'dq_{i}' for i in range(2*n_seg)]
    all_cols = y_meas_cols + y_pred_cols + q_cols + dq_cols
    Y_true_all = Y_true.reshape(-1, 6)
    Y_pred_all = Y_pred.reshape(-1, 6)
    X_pred_all = X_pred.reshape(-1, 4*n_seg)
    all_vals = np.hstack((Y_true_all, Y_pred_all, X_pred_all))

    df = pd.DataFrame(
        all_vals, 
        columns=all_cols
    )
    df.to_csv(f'evaluation/data/{dlo}_{model}_predictions_{rollout_length}steps.csv', index=False)


def how_nseg_affects_predictions(save_fig: bool = False):
    dlo1 = 'aluminium_rod'
    dlo2 = 'pool_noodle'
    dyn = 'rnn'
    dec = 'LFK'
    n_segs = [2, 5, 7, 10, 20]
    
    # Load trained models
    ar_configs = [get_model_configs(dlo1, n_seg, dyn, dec) for n_seg in n_segs]
    ar_trained_models = [load_trained_model(config) for config in ar_configs]

    pn_configs = [get_model_configs(dlo2, n_seg, dyn, dec) for n_seg in n_segs]
    pn_trained_models = [load_trained_model(config) for config in pn_configs]

    # Load test data
    window = 'sliding'
    x_rollout = 1
    ar_test_data = get_test_data(ar_configs[0], window, x_rollout)
    pn_test_data = get_test_data(pn_configs[0], window, x_rollout)

    # Evaluate models
    X_ar, Y_ar = dict(), dict()
    for n_seg, m in zip(n_segs, ar_trained_models):
        X_, Y_ = jax.vmap(m)(
            ar_test_data.U_encoder[:,0,:], 
            ar_test_data.U_dyn, 
            ar_test_data.U_decoder
        )
        X_ar[n_seg] = X_
        Y_ar[n_seg] = Y_

    X_pn, Y_pn = dict(), dict()
    for n_seg, m in zip(n_segs, pn_trained_models):
        X_, Y_ = jax.vmap(m)(
            pn_test_data.U_encoder[:,0,:], 
            pn_test_data.U_dyn, 
            pn_test_data.U_decoder
        )
        X_pn[n_seg] = X_
        Y_pn[n_seg] = Y_

    # Compute performance metrics
    ar_pos_err_norms_in_cm = dict()
    ar_vel_err_norms_in_cms = dict()
    pn_pos_err_norms_in_cm = dict()
    pn_vel_err_norms_in_cms = dict()
    for n_seg in n_segs:
        E = ar_test_data.Y - Y_ar[n_seg]
        ar_pos_err_norms_in_cm[n_seg] = 100.*nn_utils.mean_l2_norm(E[:,:,:3]).item()
        ar_vel_err_norms_in_cms[n_seg] = 100.*nn_utils.mean_l2_norm(E[:,:,3:]).item()

        E = pn_test_data.Y - Y_pn[n_seg]
        pn_pos_err_norms_in_cm[n_seg] = 100.*nn_utils.mean_l2_norm(E[:,:,:3]).item()
        pn_vel_err_norms_in_cms[n_seg] = 100.*nn_utils.mean_l2_norm(E[:,:,3:]).item()

    pos_err = [ar_pos_err_norms_in_cm, pn_pos_err_norms_in_cm]
    vel_err = [ar_vel_err_norms_in_cms, pn_vel_err_norms_in_cms]

    # Plot mean l2 norm of prediction error
    fig, axs = plt.subplots(1,2,figsize=(5,2))
    axs.reshape(-1)
    for ax, pos_err, vel_err in zip(axs, pos_err, vel_err):
        ax.plot(pos_err.keys(), list(pos_err.values()), 'o-', color='#82bfe8')
        # add second y-axis
        ax2 = ax.twinx()
        ax2.plot(vel_err.keys(), list(vel_err.values()), 'o-', color='#ff8080')
        ax.grid(alpha=0.25)
        ax.set_xlabel(r'$n_{\mathrm{el}}$')
        ax.tick_params(axis='y', labelcolor='#82bfe8')
        ax2.tick_params(axis='y', labelcolor='#ff8080')
    ax2.set_ylabel(r'$||p_\mathrm{e} - \hat p_\mathrm{e}||_2$ [cm/s]', fontdict={'color': '#ff8080'})
    axs[0].set_ylabel(r'$||p_\mathrm{e} - \hat p_\mathrm{e}||_2$ [cm]', fontdict={'color': '#82bfe8'})
    plt.tight_layout()
    plt.show()

    if save_fig:
        fig.savefig('evaluation/figures/nseg_pred_error.svg', format='svg', dpi=600, bbox_inches='tight')


def evaluate_model_performance_and_save_predictions(
        dlo: str = 'pool_noodle',# 'pool_noodle', 'aluminium_rod'
        n_seg: int = 7,
        dynamics: str = 'rnn',
        decoder: str = 'LFK',
        x_rollout: int = 1
):
    # Load trained model
    config = get_model_configs(dlo, n_seg, dynamics, decoder)
    trained_model = load_trained_model(config)

    # Load train and val data to get sacalars
    window = 'sliding'
    test_data = get_test_data(config, window, x_rollout)
    
    
    # Evaluate model
    X, Y_pred = jax.vmap(trained_model)(
        test_data.U_encoder[:,0,:], 
        test_data.U_dyn, 
        test_data.U_decoder
    )
    
    # Compute performance metrics
    E = test_data.Y - Y_pred
    pos_error_mean_l2_norm = nn_utils.mean_l2_norm(E[:,:,:3]).item()
    vel_error_mean_l2_norm = nn_utils.mean_l2_norm(E[:,:,3:]).item()

    model = f'{dynamics}_{decoder}'
    print(f"model: {model}")
    print(f"pos error: {pos_error_mean_l2_norm:.3f}")
    print(f"vel error: {vel_error_mean_l2_norm:.3f}")

    # Save predictions to csv
    save_predictions_to_csv(dlo, model, test_data.Y, Y_pred, X)


def visualize_dlo_motion(
        dlo:str = 'pool_noodle', 
        n_seg: int = 7, 
        dynamics: str = 'rnn', 
        x_rollout: int = 5
):
    # Load trained model
    decoder = 'LFK'
    config = get_model_configs(dlo, n_seg, dynamics, decoder)
    trained_model = load_trained_model(config)

    # Load train and val data to get sacalars
    window = 'sliding'
    test_data = get_test_data(config, window, x_rollout)
    panda_rlts = get_panda_rollouts(config, window, x_rollout)

    # Evaluate model
    X, Y_pred = jax.vmap(trained_model)(
        test_data.U_encoder[:,0,:], 
        test_data.U_dyn, 
        test_data.U_decoder
    )

    # Get rfem description
    trained_model.decoder._update_rfem_params()
    learned_rfem_params = trained_model.decoder.rfem_params
    model, cmodel, vmodel = models.create_setup_pinocchio_model(learned_rfem_params, add_ee_ref_joint=True)

    dt = 0.04
    n_replays = 2
    # n_window = jnp.array([1])
    for n_window in range(0, len(X)):
        q_rfem, _ = jnp.hsplit(X[n_window].squeeze(), 2)
        q_p = panda_rlts[n_window]
        q_b, _ = jnp.hsplit(test_data.U_decoder[n_window].squeeze(), 2)
        p_e = test_data.Y[n_window, :, :3]
        q = jnp.hstack((q_p, q_b, q_rfem, p_e))
        visualize_robot(np.array(q[::10,:]), dt, n_replays, model, cmodel, vmodel)



def how_rfem_regularization_affects_dlo_shape(
        dlo: str = 'pool_noodle',
        n_seg: int = 7,
        dyn_model: str = 'rnn',
):
    reg = '0.5reg'
    config_names = [f'{dlo}/PN_{dyn_model}_{n_seg}seg_LFK_{reg}.yml']
    # config_names = [f'{dlo}/PN_{dyn_model}_{n_seg}seg_LFK.yml']
    configs = get_model_configs(config_names)
    trained_models = load_trained_model(configs)

    # Get data
    train_data = get_data_used_for_training(configs[0])
    train_rollout_length = configs[0]['rollout_length']
    test_rollout_length = 10*train_rollout_length
    # n_test_trajs = configs[0]['test_trajs']
    n_test_trajs = [6]
    test_trajs = data_utils.load_trajs(n_test_trajs, configs[0]['DLO'])
    test_data = data_pp.construct_test_dataset_from_trajs(
        test_trajs, test_rollout_length, train_data, 'sliding', scale_outputs=False
    )

    # Do inference
    X, Y_preds = evaluate_models(trained_models, test_data)

    # Compute position prediction error
    pos_error_mean_l2_norm = compute_mean_l2_norm_of_pos_prediction_error(test_data.Y, Y_preds)
    vel_error_mean_l2_norm = compute_mean_l2_norm_of_vel_prediction_error(test_data.Y, Y_preds)
    print(f"pos error: {pos_error_mean_l2_norm}")
    print(f"vel error: {vel_error_mean_l2_norm}")

    # Get rfem description
    trained_models[0].decoder._update_rfem_params()
    learned_rfem_params = trained_models[0].decoder.rfem_params
    pin_dlo_model, pin_dlo_geom_model = models.create_rfem_pinocchio_model(
        learned_rfem_params, add_ee_ref_joint=False
    )

    # Visualize DLO shape
    # n_window = 6 # 6, 7, 9
    for n_window in range(0, len(test_data.Y)) :
        q_rfem, _ = jnp.hsplit(X[0][n_window], 2)
        q_b, _ = jnp.hsplit(test_data.U_decoder[n_window], 2)
        q = jnp.hstack((q_b, q_rfem))
        visualize_robot(np.asarray(q), 0.004, 3, pin_dlo_model, pin_dlo_geom_model)
    # for k in range(5, test_rollout_length, 60):
    #     q_ = jnp.tile(q[[k],:], (1000, 1))
    #     visualize_robot(np.asarray(q_), 0.004, 1, pin_dlo_model, pin_dlo_geom_model)


def plot_output_prediction(save_fig: False, x_rollout: int = 10):
    n_seg = 7
    # config_names = [f'resnet_{n_seg}seg_FFK.yml',
    #                 f'resnet_{n_seg}seg_NN.yml']
    config_names = [f'PN_rnn_{n_seg}seg_LFK.yml']
    configs = get_model_configs(config_names)
    trained_models = load_trained_model(configs)

    # Get data
    train_data = get_data_used_for_training(configs[0])
    train_rollout_length = configs[0]['rollout_length']
    try:
        n_test_trajs = configs[0]['test_trajs']
    except KeyError:
        n_test_trajs = [17]
    test_rollout_length = x_rollout*train_rollout_length
    test_trajs = data_pp.load_trajs(n_test_trajs)
    test_data = data_pp.construct_test_dataset_from_trajs(
        test_trajs, test_rollout_length, train_data, 'sliding', scale_outputs=False
    )

    # Do inference
    X_rnns, Y_preds = evaluate_models(trained_models, test_data, output_scalar=None)

    # Plot outputs
    dt = 0.004
    n_windows = jnp.array([0])
    Y_preds_traj = [Y[n_windows].reshape(-1,6) for Y in Y_preds]
    Y_test_traj = test_data.Y[n_windows].reshape(-1,6)
    t_ = jnp.arange(0, Y_test_traj.shape[0])*dt
    T = [t_]*(len(Y_preds_traj) + 1)
    Y = [Y_test_traj] + Y_preds_traj
    traj_lbls = ['meas'] + [c['name'] for c  in configs]
    t_reset = jnp.arange(0, Y_test_traj.shape[0], test_data.Y.shape[1])*dt

    # _, ax = plt.subplots(figsize=(8,4))
    # for e, l in zip(err_trajs, traj_lbls[1:]):
    #     ax.plot(t_, 100*e, label=l)
    # ax.set_xlabel(r'$t$ [s]')
    # ax.set_ylabel(r'$|p_{\mathrm{e}} - \hat p_{\mathrm{e}}|_2$ [cm]')
    # ax.grid(alpha=0.25)
    # plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    # plt.tight_layout()
    # plt.show()

    y_lbls = [r'$p_{\mathrm{e},x}$ [m]', r'$p_{\mathrm{e},y}$ [m]',
              r'$p_{\mathrm{e},z}$ [m]']
    fig, axs = plt.subplots(3, 1, figsize=(10,5), sharex=True)
    axs = axs.reshape(-1)
    for t, y, l in zip(T, Y, traj_lbls):
        for k, ax in enumerate(axs):
            if l == 'meas':
                ax.plot(t, y[:,k], 'k--', lw=1.5, label=l)
            else:
                ax.plot(t, y[:,k], lw=1, label=l)
            ax.set_ylabel(y_lbls[k])
            ax.grid(alpha=0.25)
            ax.set_xlim([0, 10])
    axs[2].set_xlabel(r'$t$ [s]')
    plt.legend(bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    plt.show()

    if save_fig:
        fig.savefig('CORL_figs/long_output_pred.pdf', format='pdf', dpi=600, bbox_inches='tight')


def compute_inference_time():
    n_seg = 7
    dyn_type = 'node'
    dec_type = 'NN'
    config_names = [f'{dyn_type}_{n_seg}seg_{dec_type}.yml']
    configs = get_model_configs(config_names)
    trained_models = load_trained_model(configs)

    # Get data
    train_data = get_data_used_for_training(configs[0])
    train_rollout_length = configs[0]['rollout_length']
    test_rollout_length = train_rollout_length
    n_test_trajs = [17]
    test_trajs = data_pp.load_trajs(n_test_trajs)
    test_data = data_pp.construct_test_dataset_from_trajs(
        test_trajs, test_rollout_length, train_data, 'sliding', scale_outputs=False
    )

    n_window = 14
    x0 = trained_models[0].encoder(test_data.U_encoder[n_window,0,:])
    X = trained_models[0].dynamics(x0, test_data.U_rnn[n_window])

    reps = 5000
    start_time = time.time()
    for _ in range(reps):
        X = trained_models[0].dynamics(x0, test_data.U_rnn[n_window])
    end_time = time.time()
    execution_time = (end_time - start_time)/reps  # Calculate the execution time
    print(f"Execution time: {execution_time*1000:.3f} ms")


def evaluate_model_performance_and_save_predictions_for_all_models(x_rollout: int = 1):
    dlos = ['aluminium_rod', 'pool_noodle']
    n_seg = 7
    dyn = ['rnn', 'resnet']
    dec = ['LFK', 'NN']
    for dlo in dlos:
        for dynamics in dyn:
            for decoder in dec:
                evaluate_model_performance_and_save_predictions(
                    dlo, n_seg, dynamics, decoder, x_rollout
                )


if __name__ == "__main__":
    # evaluate_model_performance_and_save_predictions_for_all_models(x_rollout=1)
    # evaluate_model_performance_and_save_predictions()
    # how_nseg_affects_predictions(save_fig=True)
    visualize_dlo_motion(dlo='pool_noodle')


    # plot_hidden_rfem_state_evolution()
    # analyse_encoder()
    # visualize_rfem_motion()
    # plot_output_prediction(save_fig=False)
    # how_rfem_regularization_affects_dlo_shape()
    # compute_inference_time()