import yaml
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import pandas as pd
import time


def prediction_error_l2_norm(Y_true: jnp.ndarray, Y_pred: jnp.ndarray):
    E = Y_true - Y_pred
    return jnp.sqrt(jnp.sum(E**2, axis=1))


def prediction_error_mean_l2_norm(Y_true: jnp.ndarray, Y_pred: jnp.ndarray):
    return jnp.mean(prediction_error_l2_norm(Y_true, Y_pred))


def prediction_error_batch_mean_l2_norm(Y_true: jnp.ndarray, Y_pred: jnp.ndarray):
    mean_l2_norms = jax.vmap(prediction_error_mean_l2_norm)(Y_true, Y_pred)
    return jnp.mean(mean_l2_norms)


def compute_mean_l2_norm_of_pos_prediction_error(
        Y_true: jnp.ndarray, 
        Y_preds: List[jnp.ndarray]
    ):
    out = []
    for Y_pred in Y_preds:
        out.append(
            prediction_error_batch_mean_l2_norm(Y_true[:,:,:3], Y_pred[:,:,:3]).item()
        )
    return out


def compute_mean_l2_norm_of_vel_prediction_error(
        Y_true: jnp.ndarray, 
        Y_preds: List[jnp.ndarray]
    ):
    out = []
    for Y_pred in Y_preds:
        out.append(
            prediction_error_batch_mean_l2_norm(Y_true[:,:,3:], Y_pred[:,:,3:]).item()
        )
    return out


def compute_pos_prediction_error_l2_norm(Y, Y_pred_models):
    pos_err_lengths = []
    for Y_pred in Y_pred_models:
        pos_err_lengths.append(
            prediction_error_l2_norm(Y[:,:,:3], Y_pred[:,:,:3])
        )
    return pos_err_lengths


def get_models_configs(config_names: List[str]) -> List[dict]:
    # TODO check that all configs have the same train trajs
    # otherwise scaling the test trajs is not well defined

    configs_dir = 'examples/short_aluminium_rod/experiment_configs/'
    experiment_configs_paths = [configs_dir + name
                               for name in config_names]
    # Load config files
    configs = []
    for config_path in experiment_configs_paths:
        with open(config_path, 'r') as file:
            configs.append(yaml.safe_load(file))
    return configs


def get_trained_models(configs: List[dict]) -> List:
    # Some useful function to work with NODE
    where = lambda x: x.dynamics.integrator.setting
    new_intg_setting = IntegratorSetting(
        dt=0.004, rtol=1e-6, atol=1e-8, method=IntegrationMethod.RK45
    )

    # Load mode skeletons
    model_pytrees = []
    for c in configs:
        model_pytrees.append(get_model(c))

    # Deserialoze saved trained models
    trained_models = []
    models_dir = 'examples/short_aluminium_rod/saved_models/'
    for c, m in zip(configs, model_pytrees):
        model_path_ = models_dir + c['name'] + '.eqx'
        trained_m = eqx.tree_deserialise_leaves(model_path_, m)
        # trained_m = eqx.tree_at(where, trained_m, new_intg_setting)
        trained_models.append(trained_m)

    return trained_models


def evaluate_models(models: List, data, output_scalar, batch_size: int = 256):
    def eval_minibatch(m, out_scalar, U_enc, U_dyn, U_dec):
        X_, Y_ = jax.vmap(m)(U_enc, U_dyn, U_dec)
        if out_scalar is not None:
                Y_ = out_scalar.vtransform(Y_)
        return X_, Y_

    X_rnn, Y_pred = [], []
    for m in models:
        if data.Y.shape[0] > batch_size:
            X_, Y_ = [], []
            for k in range(0, data.Y.shape[0], batch_size):
                s_idx = k
                if k + batch_size < data.Y.shape[0]:
                    e_idx = k + batch_size
                else:
                    e_idx = data.Y.shape[0]
                X_mb, Y_mb = eval_minibatch(
                    m,
                    output_scalar,
                    data.U_encoder[s_idx:e_idx,0,:],
                    data.U_rnn[s_idx:e_idx],
                    data.U_decoder[s_idx:e_idx]
                )
                X_.append(X_mb)
                Y_.append(Y_mb)
            X_rnn.append(jnp.vstack(X_))
            Y_pred.append(jnp.vstack(Y_))
        else:
            X_, Y_ = eval_minibatch(
                m,
                output_scalar, 
                data.U_encoder[:,0,:],
                data.U_rnn,
                data.U_decoder
            )
            X_rnn.append(X_)
            Y_pred.append(Y_)
    return X_rnn, Y_pred


def compute_performance_metrics(Y, Y_pred_models):
    rmse, mae = [], []
    for Y_pred in Y_pred_models:
        rmse.append(
            jnp.sqrt(mse_loss(Y, Y_pred)).item()
        )
        mae.append(
            mae_loss(Y, Y_pred).item()
        )
    return rmse, mae


def sanity_check_rollout_lengths(configs, rollout_length):
    pass


def get_data_used_for_training(config):
    val_size = 0.15
    rollout_length = config['rollout_length']
    data_key = jax.random.PRNGKey(config['data_seed'])
    train_trajs = data_prpr.load_trajs(config['train_trajs'])

    data_key, data_subkey = jax.random.split(data_key)
    train_data, val_data = data_prpr.construct_train_val_datasets_from_trajs(
        train_trajs, rollout_length, val_size, data_subkey
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
    test_trajs = data_prpr.load_trajs(n_test_trajs)
    test_data = data_prpr.construct_test_dataset_from_trajs(
        test_trajs, rollout_length, train_data, 'sliding', scale_outputs=True
    )
    output_scalar = test_data.output_scalar

    X_rnns, Y_preds = evaluate_models(trained_models, test_data, output_scalar)
    pos_error_mean_l2_norm = compute_mean_l2_norm_of_pos_prediction_error(test_data.Y, Y_preds)
    vel_error_mean_l2_norm = compute_mean_l2_norm_of_vel_prediction_error(test_data.Y, Y_preds)

    # rmse, mae = compute_performance_metrics(test_data.Y, Y_preds)
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


def plot_hidden_rfem_state_evolution():
    # Load trained models
    n_segs = [5]
    dyn_type = 'rnn'
    config_names = ([f'{dyn_type}_nseg{x}_FFK.yml' for x in n_segs] + 
                    [f'{dyn_type}_nseg{x}_LFK.yml' for x in n_segs])
    configs = get_models_configs(config_names)
    trained_models = get_trained_models(configs)

    # Load train and val data to get sacalars
    train_data = get_data_used_for_training(configs[0])
    rollout_length = configs[0]['rollout_length']
    n_test_trajs = [17] # [15, 17]
    test_trajs = data_prpr.load_trajs(n_test_trajs)
    test_data = data_prpr.construct_test_dataset_from_trajs(
        test_trajs, rollout_length, train_data, 'sliding', scale_outputs=True
    )
    output_scalar = test_data.output_scalar
    X_rnns, Y_preds = evaluate_models(trained_models, test_data, output_scalar)

    X_rnn_ffk, X_rnn_lfk = X_rnns
    plot_rfem_states(X_rnn_lfk)


def plot_numerical_velocities_against_real(X):
    dt = 0.004
    # Reshape sliding windows into a continuous vector
    X_ = X.reshape(-1, X.shape[2])
    q_rfem, dq_rfem = jnp.hsplit(X_,2)
    t_ = jnp.arange(0, X_.shape[0])*dt
    t_reset = jnp.arange(0, X_.shape[0], X.shape[1])*dt

    dq_rfem_num = (q_rfem[1:,:] - q_rfem[:-1,:])/dt

    n_jnts = [0,1]
    _, ax = plt.subplots()
    ax.plot(t_, dq_rfem[:,n_jnts])
    ax.plot(t_[1:], dq_rfem_num[:,n_jnts], ls='--')
    ax.set_ylim([-5,5])
    plt.show()


def plot_rfem_states(X):
    dt = 0.004
    # Reshape sliding windows into a continuous vector
    X_ = X.reshape(-1, X.shape[2])
    q_rfem, dq_rfem = jnp.hsplit(X_,2)
    t_ = jnp.arange(0, X_.shape[0])*dt
    t_reset = jnp.arange(0, X_.shape[0], X.shape[1])*dt

    _, axs = plt.subplots(2,1)
    axs[0].plot(t_, q_rfem)
    for tr in t_reset:
        axs[0].axvline(tr, ls='--')
    axs[0].grid(alpha=0.25)

    axs[1].plot(t_, dq_rfem)
    for tr in t_reset:
        axs[1].axvline(tr, ls='--')
    axs[1].grid(alpha=0.25)
    plt.tight_layout()
    plt.show()


def dlo_shape():
    pass


def performance_on_different_rollout_lengths():
    n_seg = 7
    dyn = 'resnet'
    config_names = [f'{dyn}_{n_seg}seg_NN.yml']#,
                    #  f'{dyn}_{n_seg}seg_NN.yml'])
    configs = get_models_configs(config_names)
    trained_models = get_trained_models(configs)

    # Load train and val data to get sacalars
    train_data = get_data_used_for_training(configs[0])
    train_rollout_length = configs[0]['rollout_length']
    test_rollout_length = 10*train_rollout_length
    n_test_trajs = [2, 15, 17] # [15, 17]
    test_trajs = data_prpr.load_trajs(n_test_trajs)
    test_data = data_prpr.construct_test_dataset_from_trajs(
        test_trajs, test_rollout_length, train_data, 'sliding', scale_outputs=True
    )
    output_scalar = test_data.output_scalar

    X_rnns, Y_preds = evaluate_models(trained_models, test_data, output_scalar, batch_size=128)
    pos_error_mean_l2_norm = compute_mean_l2_norm_of_pos_prediction_error(test_data.Y, Y_preds)
    vel_error_mean_l2_norm = compute_mean_l2_norm_of_vel_prediction_error(test_data.Y, Y_preds)

    column_names = [c['name'] for c in configs]
    row_names = ['pos_err_norm', 'vel_err_norm']
    df = pd.DataFrame(columns=column_names, index=row_names)
    df.loc['pos_err_norm'] = pos_error_mean_l2_norm
    df.loc['vel_err_norm'] = vel_error_mean_l2_norm
    pd.set_option('display.precision', 3)
    print(df)



def analyse_encoder():
    n_seg = 7
    dyn_type = 'resnet'
    config_names = [f'{dyn_type}_{n_seg}seg_NN.yml']
    configs = get_models_configs(config_names)
    trained_models = get_trained_models(configs)

    # Load train and val data to get sacalars
    train_data = get_data_used_for_training(configs[0])
    rollout_length = configs[0]['rollout_length']
    n_test_trajs = [17] # [15, 17]
    test_trajs = data_prpr.load_trajs(n_test_trajs)
    test_data = data_prpr.construct_test_dataset_from_trajs(
        test_trajs, rollout_length, train_data, 'sliding', scale_outputs=True
    )
    output_scalar = test_data.output_scalar
    X_rnns, Y_preds = evaluate_models(trained_models, test_data, output_scalar)
    X_encs, Y_encs = [], []
    for m in trained_models:
        X_ = jax.vmap(jax.vmap(m.encoder))(test_data.U_encoder)
        X_encs.append(X_)
        Y_encs.append(jax.vmap(jax.vmap(m.decoder))(X_, test_data.U_decoder))


    t = jnp.arange(0, rollout_length+1)*0.004
    Q_rfem_enc, dQ_rfem_enc = jnp.split(X_encs[0], 2, axis=2)
    Q_rfem_dyn, dQ_rfem_dyn = jnp.split(X_rnns[0], 2, axis=2)
    n_window = 8
    n_joints = [2,3]
    _, axs = plt.subplots(2,1)
    axs[0].plot(t, Q_rfem_enc[n_window,:,n_joints].T)
    axs[0].plot(t, Q_rfem_dyn[n_window,:,n_joints].T, '--')

    axs[1].plot(t, dQ_rfem_enc[n_window,:,n_joints].T)
    axs[1].plot(t, dQ_rfem_dyn[n_window,:,n_joints].T, '--')
    plt.tight_layout()

    T = [t]*3
    Y = [test_data.Y[n_window], Y_encs[0][n_window], Y_preds[0][n_window]]
    traj_lbls = ['meas', 'enc', 'dyn']
    axes = ['x', 'y', 'z']
    y_lbls = ([f'p_e_{x}' for x in axes] + [f'dp_e_{x}' for x in axes])
    _, axs = plt.subplots(3, 2, sharex=True)
    axs = axs.T.reshape(-1)
    for t, y, l in zip(T, Y, traj_lbls):
        for k, ax in enumerate(axs):
            ax.plot(t, y[:,k], label=l)
            ax.set_ylabel(y_lbls[k])
            ax.grid(alpha=0.25)
            ax.legend()
    plt.tight_layout()
    plt.show()


def visualize_rfem_motion():
    n_seg = 7
    dyn_type = 'resnet'
    config_names = [f'{dyn_type}_{n_seg}seg_FFK.yml',
                    f'{dyn_type}_{n_seg}seg_NN.yml']
    configs = get_models_configs(config_names)
    trained_models = get_trained_models(configs)

    # Get data
    train_data = get_data_used_for_training(configs[0])
    rollout_length = configs[0]['rollout_length']
    n_test_trajs = [17]
    test_trajs = data_prpr.load_trajs(n_test_trajs)
    test_data = data_prpr.construct_test_dataset_from_trajs(
        test_trajs, 5*rollout_length, train_data, 'sliding', scale_outputs=False
    )

    # Do inference
    X_rnns, Y_preds = evaluate_models(trained_models, test_data, output_scalar=None)

    pin_dlo_model, pin_dlo_geom_model = models.create_rfem_pinocchio_model(
        trained_models[0].decoder.rfem_params, add_ee_ref_joint=True
    )

    n_windows = jnp.array([2, 3])
    for x_rnn_FK, y_NN, u_dec, y in zip(
        X_rnns[0][n_windows], Y_preds[1][n_windows], test_data.U_decoder[n_windows], test_data.Y[n_windows]):
        q_rfem, _ = jnp.hsplit(x_rnn_FK, 2)
        q_b, _ = jnp.hsplit(u_dec, 2)
        pe_meas, _ = jnp.hsplit(y, 2)
        pe_pred_NN, _ = jnp.hsplit(y_NN, 2)
        q = jnp.hstack((q_b, q_rfem, pe_meas, pe_pred_NN))
        visualize_robot(np.asarray(q), 0.004, 4, pin_dlo_model, pin_dlo_geom_model)


def how_rfem_regularization_affects_dlo_shape():
    n_seg = 7
    dyn_type = 'rnn'
    reg = '_1reg'
    config_names = [f'{dyn_type}_{n_seg}seg_FFK{reg}.yml']
    configs = get_models_configs(config_names)
    trained_models = get_trained_models(configs)

    # Get data
    train_data = get_data_used_for_training(configs[0])
    rollout_length = configs[0]['rollout_length']
    n_test_trajs = [17]
    test_trajs = data_prpr.load_trajs(n_test_trajs)
    test_data = data_prpr.construct_test_dataset_from_trajs(
        test_trajs, 1*rollout_length, train_data, 'sliding', scale_outputs=False
    )

    # Do inference
    X_rnns, Y_preds = evaluate_models(trained_models, test_data, output_scalar=None)

    pin_dlo_model, pin_dlo_geom_model = models.create_rfem_pinocchio_model(
        trained_models[0].decoder.rfem_params, add_ee_ref_joint=False
    )

    n_window = 14
    q_rfem, _ = jnp.hsplit(X_rnns[0][n_window], 2)
    q_b, _ = jnp.hsplit(test_data.U_decoder[n_window], 2)
    q = jnp.hstack((q_b, q_rfem))
    for k in range(0, rollout_length, 62):
        q_ = jnp.tile(q[[k],:], (1000, 1))
        visualize_robot(np.asarray(q_), 0.004, 1, pin_dlo_model, pin_dlo_geom_model)


def plot_output_prediction(save_fig: False):
    n_seg = 7
    config_names = [f'resnet_{n_seg}seg_FFK.yml',
                    f'resnet_{n_seg}seg_NN.yml']
    configs = get_models_configs(config_names)
    trained_models = get_trained_models(configs)

    # Get data
    train_data = get_data_used_for_training(configs[0])
    train_rollout_length = configs[0]['rollout_length']
    n_test_trajs = [17]
    test_rollout_length = 10*train_rollout_length
    test_trajs = data_prpr.load_trajs(n_test_trajs)
    test_data = data_prpr.construct_test_dataset_from_trajs(
        test_trajs, test_rollout_length, train_data, 'sliding', scale_outputs=False
    )

    # Do inference
    X_rnns, Y_preds = evaluate_models(trained_models, test_data, output_scalar=None)
    pos_err_lengths = compute_pos_prediction_error_l2_norm(test_data.Y, Y_preds)

    # Plot outputs
    dt = 0.004
    n_windows = jnp.array([2])
    Y_preds_traj = [Y[n_windows].reshape(-1,6) for Y in Y_preds]
    err_trajs = [err[n_windows].reshape(-1,1) for err in pos_err_lengths]
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
                ax.plot(t, y[:,k], 'k', lw=2.5, label=l)
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
    test_trajs = data_prpr.load_trajs(n_test_trajs)
    test_data = data_prpr.construct_test_dataset_from_trajs(
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
    # how_rfem_regularization_affects_dlo_shape()
    compute_inference_time()