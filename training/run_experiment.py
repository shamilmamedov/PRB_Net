import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
import jax
import optax
import wandb
import yaml
from typing import Union, List
import fire
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"

from FEIN import rnn, resnet, neural_ode, encoders, decoders
import FEIN.utils.nn as nn_utils
import FEIN.utils.ode as ode_utils
import FEIN.utils.data as data_utils
import FEIN.utils.kinematics as kin_utils
import FEIN.rfem_kinematics.models as rfem_models
import training.preprocess_data as data_prpr


class DLOModel(eqx.Module):
    n_seg: int
    encoder: eqx.Module
    dynamics: eqx.Module
    decoder: eqx.Module

    def __init__(self, n_seg, encoder, dynamics, decoder) -> None:
        self.n_seg = n_seg
        self.encoder = encoder
        self.dynamics = dynamics
        self.decoder = decoder

    @eqx.filter_jit
    def __call__(self, u_enc, U_dyn, U_dec):
        x_0 = self.encoder(u_enc)
        X = self.dynamics(x_0, U_dyn)
        Y = jax.vmap(self.decoder, in_axes=(0,0))(X, U_dec)
        return X, Y    


def get_encoder(enc_configs, n_seg, key):
    """ Based on the encoder configs creates an instance of the
    specified encoder"""

    type_to_class = {
        'NN': encoders.NNEncoder,
        'PINN': encoders.PIEncoder
    }
    type_in_out_sizes = {
        'NN': (18, 4*n_seg),
        'PINN': (9, 2*n_seg)
    }
    if enc_configs['type'] in type_to_class:
        in_size, out_size = type_in_out_sizes[enc_configs['type']]
        mlp_params = nn_utils.MLPParameters(
            in_size=in_size,
            out_size=out_size,
            width_size=enc_configs['width'],
            depth=enc_configs['depth'],
            activation=nn_utils.activations[enc_configs['activation']] 
        )
        return type_to_class[enc_configs['type']](mlp_params, key=key)
    else:
        raise ValueError


def get_dynamics(dyn_configs, n_seg, key):
    """ Supported dynamics models are RNN, ResNet, DiscretizedNODe
    """
    n_x = 4*n_seg
    if dyn_configs['type'] == 'RNN':
        dyn_params = rnn.RNNParameters(
            input_size=dyn_configs['n_u'],
            hidden_size=n_x
        )
        return rnn.RNN(dyn_params, key=key)
    else:
        dyn_params = nn_utils.MLPParameters(
            in_size=dyn_configs['n_u'] + n_x,
            out_size=n_x,
            width_size=dyn_configs['width'],
            depth=dyn_configs['depth'],
            activation=nn_utils.activations[dyn_configs['activation']]
        )
        if dyn_configs['type'] == 'ResNet':
            return resnet.ResNet(dyn_params, key=key)
        else:
            dyn_params.out_size = n_x//2
            node_ = neural_ode.SONODE(dyn_params, key=key)
            intg_setting = ode_utils.IntegratorSetting(
                dt=dyn_configs['integrator']['dt'],
                rtol=dyn_configs['integrator']['rtol'],
                atol=dyn_configs['integrator']['atol'],
                method=ode_utils.IntegrationMethod.RK45
            )
            integrator_ = neural_ode.Integrator(intg_setting)
            return neural_ode.DiscretizedNODE(node_, integrator_)


def get_decoder(dec_configs, dlo, n_seg, key):
    """ Supported decoders are NN, FK and Trainable FK
    """
    if dec_configs['type'] == 'NN':
        dec_params = nn_utils.MLPParameters(
            in_size=12+4*n_seg,
            out_size=dec_configs['n_y'],
            width_size=dec_configs['width'],
            depth=dec_configs['depth'],
            activation=nn_utils.activations[dec_configs['activation']] 
        )
        return decoders.NNDecoder(n_seg, dec_params, key=key)
    else:
        configs_dir = 'FEIN/rfem_kinematics/configs/'
        if dlo == 'aluminium-rod':
            marker_pos_path = configs_dir + 'alrod-vicon-marker-locations.yaml'
            dlo_phys_params_path = configs_dir + 'alrod-physical-params.yaml'
        elif dlo == 'pool-noodle':
            marker_pos_path = configs_dir + 'pool-noodle-vicon-marker-locations.yaml'
            dlo_phys_params_path = configs_dir + 'pool-noodle-physical-params.yaml'
        else:
            raise ValueError('Please specify a valid DLO name') 

        bjoint = kin_utils.JointType.FREE
        dlo_params = rfem_models.load_dlo_params_from_yaml(dlo_phys_params_path)
        px_markers = data_utils.load_vicon_marker_locations(marker_pos_path)
        pe_marker = jnp.array([[px_markers['p_be'], 0., 0.]]).T
        dec_params = rfem_models.RFEMParameters(n_seg, dlo_params, bjoint, [pe_marker])
        if dec_configs['type'] == 'FK':
            return decoders.FKDecoder(dec_params)
        else:
            return decoders.TrainableFKDecoder(dec_params)


def get_model(configs):
    dlo = configs['DLO']
    n_seg = configs['n_seg']
    model_key = jax.random.PRNGKey(configs['model_seed'])
    enc_key, dyn_key, dec_key = jrandom.split(model_key, 3)

    # Parse, Create and instantiate an encoder
    encoder = get_encoder(configs['encoder'], n_seg, enc_key)
    dynamics = get_dynamics(configs['dynamics'], n_seg, dyn_key)
    decoder = get_decoder(configs['decoder'], dlo, n_seg, dec_key)
    return DLOModel(n_seg, encoder, dynamics, decoder)

    
def get_optimizer(configs, steps_per_epoch):
    @optax.inject_hyperparams
    def set_learning_rate(learning_rate):
        if opt_name == 'adamw':
            wd = configs['weight_decay']
            return optimizer_fcn(learning_rate=learning_rate, weight_decay=wd)
        else:
            return optimizer_fcn(learning_rate=schedule)

    optimizer_optax = {
        'adam': optax.adam,
        'adamw': optax.adamw,
        'rmsprop': optax.rmsprop,
        'sgd': optax.sgd,
        'adabelief': optax.adabelief
    }
    transition_epochs = configs['transition_epochs']
    lr_init = configs['initial_learning_rate']

    # Scheduler
    if configs['learning_rate_scheduler'] == 'constant':
        schedule = lr_init
    else:
        lr_end = configs['final_learning_rate']
        power = configs['polynomial_power']
        schedule = optax.polynomial_schedule(
            init_value=lr_init,
            end_value=lr_end,
            power=power,
            transition_steps=transition_epochs*steps_per_epoch,
            transition_begin=0
        )

    # Optimizer
    opt_name = configs['optimizer']
    optimizer_fcn = optimizer_optax[opt_name]
    optim = set_learning_rate(schedule)
    return optim


def get_rollout_length_scheduler(config):
    try:
        c_scheduler = config['rollout_length_scheduler']
        init_value = c_scheduler['init_value']
        boundaries = c_scheduler['boundaries']
        scales = c_scheduler['scales']
        boundaries_and_scales = dict(zip(boundaries, scales))
        return optax.piecewise_constant_schedule(
                    init_value,
                    boundaries_and_scales
                )
    except KeyError:
        return optax.constant_schedule(1.)


def get_data(config, key):
    rollout_length = config['rollout_length']

    if config['DLO'] == 'aluminium-rod':
        val_size = 0.15
        trajs = data_utils.load_trajs(config['train_trajs'], config['DLO'])
        train_trajs, val_trajs = data_prpr.split_trajs_into_train_val(trajs, val_size, key)
    elif config['DLO'] == 'pool-noodle':
        train_trajs = data_utils.load_trajs(config['train_trajs'], config['DLO'])
        val_trajs = data_utils.load_trajs(config['val_trajs'], config['DLO'])
    else:
        raise ValueError('Please specify a valid DLO in the config file')

    train_data, val_data = data_prpr.construct_train_val_datasets(
        train_trajs, val_trajs, rollout_length
    )
    return train_data, val_data


def main(config: Union[str, List] = None, wandb_mode: str = 'online', save_model: bool = True):
    @eqx.filter_jit
    def compute_loss(model, U_enc, U_dyn, U_dec, Y):
        X_dyn, Y_pred_dyn = jax.vmap(model, in_axes=(0,0,0))(U_enc[:,0,:], U_dyn, U_dec)

        Y_pred_dyn_scaled = output_scalar.vtransform(Y_pred_dyn)
        output_prediction_loss_dyn = nn_utils.weighted_mse_loss(
            Y, Y_pred_dyn_scaled, w_y, w_t[:Y.shape[1],:]
        )
        
        try: 
            act_ = jnp.sum(model.decoder.rfem_lengths_sqrt**2) - model.decoder.rod_length
            length_loss = nn_utils.l2_loss(act_, alpha_dlo_length)
            delta_rfem_meshing = model.decoder.rfem_lengths_sqrt - rfem_meshing_sqrt
            rfem_legth_loss = nn_utils.l2_loss(delta_rfem_meshing, alpha_rfem_length)

            delta_marker_pos = p_marker_calibration - model.decoder.p_marker
            marker_pos_loss = nn_utils.l2_loss(delta_marker_pos, alpha_p_marker)

            qb_offset_loss = (nn_utils.l2_loss(model.decoder.qb_offset[:3], alpha_p_b) + 
                              nn_utils.l2_loss(model.decoder.qb_offset[3:], alpha_phi_b))
        except AttributeError:
            length_loss = 0.
            qb_offset_loss = 0.
            rfem_legth_loss = 0.
            marker_pos_loss = 0.

        n_q = 2*model.n_seg
        q_rfem_loss_dyn = nn_utils.l2_loss(X_dyn[:,:,:n_q], alpha_q_rfem)
        dq_rfem_loss_dyn = nn_utils.l2_loss(X_dyn[:,:,n_q:], alpha_dq_rfem) 
        loss = (output_prediction_loss_dyn + q_rfem_loss_dyn + dq_rfem_loss_dyn + 
                length_loss + qb_offset_loss + rfem_legth_loss + marker_pos_loss)
        return loss
    

    @eqx.filter_jit
    def make_step(model, U_enc, U_dyn, U_dec, Y, opt_state):
        loss, grads = eqx.filter_value_and_grad(compute_loss)(model, U_enc, U_dyn, U_dec, Y)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return loss, grads, model, opt_state


    # Load config file
    if config is None:
        run = wandb.init(allow_val_change=True)
        config = wandb.config
    else:
        with open(config, 'r') as file:
            config = yaml.safe_load(file)
    
        # Init weight and biases
        run = wandb.init(
                project='learning-dlo-dynamics-with-rnn',
                job_type='experiment-better-encoder-decoder',
                config=config,
                name=config['name'],
                mode=wandb_mode # 'online' 'disabled'
        ) 
        print(config['name'])


    # Get data
    batch_size = config['batch_size']
    rollout_length = config['rollout_length']
    data_key = jax.random.PRNGKey(config['data_seed']) 

    data_key, data_subkey = jax.random.split(data_key)
    train_data, val_data = get_data(config, data_subkey)
    output_scalar = train_data.output_scalar

    train_data_loader = data_utils.DLODataLoader(
        train_data.U_encoder,
        train_data.U_dyn,
        train_data.U_decoder,
        train_data.Y,
        data_key   
    )
    rollout_length_scheduler = get_rollout_length_scheduler(config)

    # Get model
    model = get_model(config)


    # Setup optimizer
    n_epochs = config['n_epochs']
    steps_per_epoch = int(len(train_data_loader)/batch_size)
    optim = get_optimizer(config, steps_per_epoch)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))


    # Parse loss weights
    alpha_q_rfem = config['q_rfem_l2']
    alpha_dq_rfem = config['dq_rfem_l2']
    alpha_p_b = 0.5
    alpha_phi_b = 0.1
    alpha_rfem_length = 0.025
    alpha_dlo_length = 1.
    alpha_p_marker = 0.5
    w_y = jnp.array([2., 2., 2., 1., 1., 1.])
    w_t = jnp.ones((rollout_length+1,1))
    w_t = w_t.at[jnp.array([0,1,2,3,4])].set(jnp.array([[5,4,3,2,1]]).T)
    try:
        rfem_meshing_sqrt = jnp.copy(model.decoder.rfem_lengths_sqrt)
        p_marker_calibration = jnp.copy(model.decoder.p_marker)
    except AttributeError:
        pass

    # Train model
    best_model = None
    best_avg_val_rmse = 1000
    print(f'Steps per epoch: {steps_per_epoch}')
    for epoch in range(n_epochs):
        lr_epoch = [] 
        epoch_train_loss = []
        for step in range(steps_per_epoch): 
            # Get current batch
            U_enc, U_dyn, U_dec, Y = train_data_loader.get_batch(batch_size)
            l_ = int(rollout_length*rollout_length_scheduler(epoch))
            loss, grads, model, opt_state = make_step(
                model, U_enc[:,:l_,:], U_dyn[:,:l_-1,:], U_dec[:,:l_,:], Y[:,:l_,:], opt_state
            )
            epoch_train_loss.append(loss.item())
            lr_epoch.append(opt_state.hyperparams['learning_rate'].item())
            
        avg_loss_ = sum(epoch_train_loss) / steps_per_epoch
        avg_lr_ = sum(lr_epoch) / steps_per_epoch
        print(f"Epoch={epoch + 1}, train loss = {avg_loss_:.5f}, lr = {avg_lr_:.5f}")

	    # Compute test set metrics
        len_val_data = len(val_data.Y)
        test_batch_size = 256
        loss_, mae_, rmse_, mse_ = [], [], [], []
        for k in range(0, len_val_data, test_batch_size):
            start_idx = k
            end_idx = k+test_batch_size
            if end_idx > len_val_data-1:
                end_idx = len_val_data-1

            U_enc_k = val_data.U_encoder[start_idx:end_idx]
            U_dyn_k = val_data.U_dyn[start_idx:end_idx]
            U_dec_k = val_data.U_decoder[start_idx:end_idx]
            Y_k = val_data.Y[start_idx:end_idx]
            loss_.append(compute_loss(model, U_enc_k, U_dyn_k, U_dec_k, Y_k).item())

            X_test_dyn_k, Y_test_pred_k = jax.vmap(model)(U_enc_k[:,0,:], U_dyn_k, U_dec_k)
            Y_test_pred_k = output_scalar.vtransform(Y_test_pred_k)
            X_test_enc_k = jax.vmap(jax.vmap(model.encoder))(U_enc_k)
            mae_.append(nn_utils.mae_loss(Y_k, Y_test_pred_k).item())
            rmse_.append(jnp.sqrt(nn_utils.mse_loss(Y_k, Y_test_pred_k)).item())
            mse_.append(nn_utils.mse_loss(X_test_dyn_k, X_test_enc_k).item())

        avg_tes_loss_ = sum(loss_) / len(loss_)   
        avg_rmse_ = sum(rmse_) / len(rmse_)
        avg_mae_ =  sum(mae_) / len(mae_)
        avg_mse_ = sum(mse_) / len(mse_)
        print(f"\t val loss = {avg_tes_loss_:.4f}")
        print(f"\t val mae = {avg_mae_:.4f}")
        print(f"\t val rmse = {avg_rmse_:.4f}")
        print(f"\t val mse Xenc-Xdyn = {avg_mse_:.4f}")
        if config['decoder']['type'] == 'TrainableFKDecoder':
            print(f'\t pb offset = {model.decoder.qb_offset[:3]}')
            print(f'\t phib offset = {model.decoder.qb_offset[3:]}')
            print(f'\t pe_marker = {model.decoder.p_marker.T}')

        if avg_mse_ > 5000:
            break
        
        if avg_rmse_ < best_avg_val_rmse:
            best_avg_val_rmse = avg_rmse_
            best_model = model    

        # Log required metrics to wandb
        log_dict = {
            'loss_train': avg_loss_,
            'loss_val': avg_tes_loss_,
            'lr': avg_lr_,
            'rmse_val': avg_rmse_,
            'mae_val': avg_mae_,
            'mse_enc_dyn_val': avg_mse_
        }
        wandb.log(log_dict)

    if save_model:
        try:
            dir2save_ = 'training/saved_models/'
            path_ = dir2save_ + config['name'] + '.eqx'
            eqx.tree_serialise_leaves(path_, best_model)
            wandb.save(path_)
        except KeyError:
            pass
    
    wandb.finish()


if __name__ == '__main__': 
    fire.Fire(main)
    # config  = 'tests/experiment_configs/rnn_test_6.yml'
    # main(config=config, wandb_mode='disabled', save_model=False)