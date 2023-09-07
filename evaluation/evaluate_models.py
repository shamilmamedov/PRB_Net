import yaml
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import pandas as pd
import time
import seaborn as sns


from training import preprocess_data as data_pp
from training.run_experiment import get_model
import FEIN.utils.nn as nn_utils
import FEIN.utils.data as data_utils
from FEIN.rfem_kinematics import models
from FEIN.rfem_kinematics.visualization import visualize_robot


def compute_mean_l2_norm_of_pos_prediction_error(
        Y_true: jnp.ndarray, 
        Y_preds: List[jnp.ndarray]
    ):
    out = []
    for Y_pred in Y_preds:
        out.append(
            nn_utils.mean_l2_norm(Y_true[:,:,:3] - Y_pred[:,:,:3]).item()
        )
    return out


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


def compute_mean_l2_norm_of_vel_prediction_error(
        Y_true: jnp.ndarray, 
        Y_preds: List[jnp.ndarray]
    ):
    out = []
    for Y_pred in Y_preds:
        out.append(
            nn_utils.mean_l2_norm(Y_true[:,:,3:] - Y_pred[:,:,3:]).item()
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


def get_models_configs(config_names: List[str]) -> List[dict]:
    # TODO check that all configs have the same train trajs
    # otherwise scaling the test trajs is not well defined

    configs_dir = 'training/experiment_configs/'
    experiment_configs_paths = [configs_dir + name
                               for name in config_names]
    # Load config files
    configs = []
    for config_path in experiment_configs_paths:
        with open(config_path, 'r') as file:
            configs.append(yaml.safe_load(file))
    return configs


def get_trained_models(configs: List[dict]) -> List:
    # Load mode skeletons
    model_pytrees = []
    for c in configs:
        model_pytrees.append(get_model(c))

    # Deserialoze saved trained models
    trained_models = []
    models_dir = 'training/saved_models/'
    for c, m in zip(configs, model_pytrees):
        model_path_ = models_dir + c['name'] + '.eqx'
        trained_m = eqx.tree_deserialise_leaves(model_path_, m)
        trained_models.append(trained_m)

    return trained_models


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


def get_data_used_for_training(config):
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


def how_nseg_affects_predictions(save_fig: bool = False):
    # Load trained models
    n_segs = [2, 5, 7, 10]
    config_names = [f'rnn_{x}seg_FFK.yml' for x in n_segs]
    configs = get_models_configs(config_names)
    trained_models = get_trained_models(configs)
    
    # Load train and val data to get sacalars
    train_data = get_data_used_for_training(configs[0])

    rollout_length = configs[0]['rollout_length']
    n_test_trajs = [2, 15, 17]
    test_trajs = data_pp.load_trajs(n_test_trajs)
    test_data = data_pp.construct_test_dataset_from_trajs(
        test_trajs, rollout_length, train_data, 'sliding', scale_outputs=True
    )
    output_scalar = test_data.output_scalar
    
    X_rnns, Y_preds = evaluate_models(trained_models, test_data, output_scalar)
    pos_error_mean_l2_norm = compute_mean_l2_norm_of_pos_prediction_error(test_data.Y, Y_preds)
    vel_error_mean_l2_norm = compute_mean_l2_norm_of_vel_prediction_error(test_data.Y, Y_preds)

    print(f"test data size: {test_data.Y.shape[0]}")
    print(f"rmse: {pos_error_mean_l2_norm}")
    print(f"mae: {vel_error_mean_l2_norm}")

    fig, ax = plt.subplots(figsize=(4,2))
    ax.plot(n_segs, 100*jnp.array(pos_error_mean_l2_norm),
            'o-', label=r'$|p_\mathrm{e} - \hat p_\mathrm{e}|_2$ [cm]')
    ax.plot(n_segs, 100*jnp.array(vel_error_mean_l2_norm), 
            'o-', label=r'$|\dot p_\mathrm{e} - \dot{\hat p_\mathrm{e}}|_2$ [cm/s]')
    ax.grid(alpha=0.25)
    ax.set_xlabel(r'$n_{\mathrm{seg}}$')
    ax.set_ylabel(r'$|\mathrm{error}|_2$')
    ax.legend()
    plt.tight_layout()
    plt.show()

    if save_fig:
        fig.savefig('CORL_figs/pred_vs_nseg.pdf', format='pdf', dpi=600, bbox_inches='tight')


def performance_on_different_rollout_lengths(
        dlo: str = 'pool_noodle',
        n_seg: int = 7,
        dyn_model: str = 'rnn',
        x_rollout: int = 1
):
    # config_names = ([f'{dyn_model}_{n_seg}seg_NN.yml',
    #                 f'{dyn_model}_{n_seg}seg_FFK.yml',
    #                 f'{dyn_model}_{n_seg}seg_LFK.yml'])
    # config_names = [f'PN_{dyn_model}_{n_seg}seg_LFK.yml']
    config_names = [f'{dlo}/PN_{dyn_model}_{n_seg}seg_LFK.yml',
                    f'{dlo}/PN_{dyn_model}_{n_seg}seg_NN.yml']
    configs = get_models_configs(config_names)
    trained_models = get_trained_models(configs)

    # Load train and val data to get sacalars
    train_data = get_data_used_for_training(configs[0])
    train_rollout_length = configs[0]['rollout_length']
    test_rollout_length = x_rollout * train_rollout_length
    n_test_trajs = configs[0]['test_trajs']
    test_trajs = data_utils.load_trajs(n_test_trajs, 'pool-noodle')
    test_data = data_pp.construct_test_dataset_from_trajs(
        test_trajs, test_rollout_length, train_data, 'sliding', scale_outputs=False
    )

    # Evaluate models
    X, Y_preds = evaluate_models(trained_models, test_data)
    
    pos_error_mean_l2_norm = compute_mean_l2_norm_of_pos_prediction_error(test_data.Y, Y_preds)
    vel_error_mean_l2_norm = compute_mean_l2_norm_of_vel_prediction_error(test_data.Y, Y_preds)

    pos_error_l2_norm = compute_l2_norm_of_pos_prediction_error(test_data.Y, Y_preds)

    column_names = [c['name'] for c in configs]
    row_names = ['pos_err_norm', 'vel_err_norm']
    df = pd.DataFrame(columns=column_names, index=row_names)
    df.loc['pos_err_norm'] = pos_error_mean_l2_norm
    df.loc['vel_err_norm'] = vel_error_mean_l2_norm
    pd.set_option('display.precision', 3)
    print(df)

    pos_error_l2_norm_in_cm = [100*err for err in pos_error_l2_norm]
    # plot the violinplot using seaborn library, use log scale along y-axis
    sns.violinplot(data=pos_error_l2_norm_in_cm)#, scale='count', inner='quartile')
    plt.yscale('log')
    plt.xticks([0, 1], ['LFK', 'NN'])
    plt.ylabel(r'$|p_\mathrm{e} - \hat p_\mathrm{e}|_2$ [cm]')
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()


def visualize_rfem_motion(n_seg: int = 7, dyn_model: str = 'rnn', x_rollout: int = 5):
    # config_names = [f'{dyn_type}_{n_seg}seg_FFK.yml',
    #                 f'{dyn_type}_{n_seg}seg_NN.yml']
    config_names = [f'PN_{dyn_model}_{n_seg}seg_LFK.yml']
    configs = get_models_configs(config_names)
    trained_models = get_trained_models(configs)

    # Get data
    train_data = get_data_used_for_training(configs[0])
    train_rollout_length = configs[0]['rollout_length']
    test_rollout_length = x_rollout * train_rollout_length
    try:
        n_test_trajs = configs[0]['test_trajs']
    except KeyError:
        n_test_trajs = [17]
    test_trajs = data_pp.load_trajs(n_test_trajs)
    test_data = data_pp.construct_test_dataset_from_trajs(
        test_trajs, test_rollout_length, train_data, 'sliding', scale_outputs=False
    )

    # Do inference
    X_rnns, Y_preds = evaluate_models(trained_models, test_data, output_scalar=None)

    # Get rfem description
    trained_models[0].decoder._update_rfem_params()
    learned_rfem_params = trained_models[0].decoder.rfem_params
    pin_dlo_model, pin_dlo_geom_model = models.create_rfem_pinocchio_model(
        learned_rfem_params, add_ee_ref_joint=False
    )

    n_window = jnp.array([1])
    q_rfem, _ = jnp.hsplit(X_rnns[0][n_window].squeeze(), 2)
    q_b, _ = jnp.hsplit(test_data.U_decoder[n_window].squeeze(), 2)
    q = jnp.hstack((q_b, q_rfem))
    visualize_robot(np.asarray(q), 0.004, 4, pin_dlo_model, pin_dlo_geom_model)

    # n_windows = jnp.array([2, 3])
    # for x_rnn_FK, y_NN, u_dec, y in zip(
    #     X_rnns[0][n_windows], Y_preds[1][n_windows], test_data.U_decoder[n_windows], test_data.Y[n_windows]):
    #     q_rfem, _ = jnp.hsplit(x_rnn_FK, 2)
    #     q_b, _ = jnp.hsplit(u_dec, 2)
    #     pe_meas, _ = jnp.hsplit(y, 2)
    #     pe_pred_NN, _ = jnp.hsplit(y_NN, 2)
    #     q = jnp.hstack((q_b, q_rfem, pe_meas, pe_pred_NN))
    #     visualize_robot(np.asarray(q), 0.004, 4, pin_dlo_model, pin_dlo_geom_model)


def how_rfem_regularization_affects_dlo_shape(
        dlo: str = 'pool_noodle',
        n_seg: int = 7,
        dyn_model: str = 'rnn',
):
    reg = '0.5reg'
    config_names = [f'{dlo}/PN_{dyn_model}_{n_seg}seg_LFK_{reg}.yml']
    # config_names = [f'{dlo}/PN_{dyn_model}_{n_seg}seg_LFK.yml']
    configs = get_models_configs(config_names)
    trained_models = get_trained_models(configs)

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
    configs = get_models_configs(config_names)
    trained_models = get_trained_models(configs)

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
    configs = get_models_configs(config_names)
    trained_models = get_trained_models(configs)

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



if __name__ == "__main__":
    # main()
    # how_nseg_affects_predictions(save_fig=True)
    # plot_hidden_rfem_state_evolution()
    # performance_on_different_rollout_lengths()
    # analyse_encoder()
    # visualize_rfem_motion()
    # plot_output_prediction(save_fig=False)
    how_rfem_regularization_affects_dlo_shape()
    # compute_inference_time()