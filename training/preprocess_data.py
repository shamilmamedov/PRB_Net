import jax.numpy as jnp
import jax
import itertools


from FEIN.utils.data import JAXMinMaxScalar, DLODataset, DLODatasetScalars


def train_val_split_idxs(len_traj, t_exctn, ratio, key):
    """ It is assumed that the trajectory is divided into two
    parts: the first part is controlled, the second part is
    a free vibrations after the robot arms has stopped

    NOTE it is assumed that the samplng time is 0.004
    
    :param traj: trajectory as jnp.array
    :param t_exctn: approximate duration of the excitation
    :param ratio: split ratio
    :param key: a key for jax random

    :return: a list of train idxs, and a list of test idxs
    """
    first_part_length = int(t_exctn[1]/0.004)
    val_length_1 = int(first_part_length*ratio)
    val_length_2 = int((len_traj-first_part_length)*ratio)

    max_start_idx_1 = first_part_length - val_length_1
    max_start_idx_2 = len_traj - val_length_2

    min_start_idx_1 = int(t_exctn[0]/0.004)
    min_start_idx_2 = first_part_length

    key, subkey = jax.random.split(key, 2)
    start_idx_1 = jax.random.randint(
        key, (1,), minval=min_start_idx_1, maxval=max_start_idx_1
    ).item()
    start_idx_2 = jax.random.randint(
        subkey, (1,), minval=min_start_idx_2, maxval=max_start_idx_2
    ).item()

    val_idxs = [
        jnp.arange(start_idx_1, start_idx_1+val_length_1),
        jnp.arange(start_idx_2, start_idx_2+val_length_2)
    ]

    train_idxs = [
        jnp.arange(0, start_idx_1),
        jnp.arange(start_idx_1+val_length_1, first_part_length),
        jnp.arange(first_part_length, start_idx_2),
        jnp.arange(start_idx_2+val_length_2, len_traj)
    ]
    return train_idxs, val_idxs


def get_train_val_idxs(trajs, val_size, key):
    traj_texctn = get_traj_to_excitation_time_dict()
    train_idxs, val_idxs = [], []
    for traj in trajs:
        t_exctn = traj_texctn[traj.n_traj]
        key, subkey = jax.random.split(key)
        train_idxs_k, val_idxs_k = train_val_split_idxs(len(traj), t_exctn, val_size, subkey)
        train_idxs.append(train_idxs_k)
        val_idxs.append(val_idxs_k)
    return train_idxs, val_idxs


def get_train_val_idxs2(n_trajs, val_size, key):
    traj_texctn = get_traj_to_excitation_time_dict()
    train_idxs, val_idxs = [], []
    for n_traj in n_trajs:
        t_exctn = traj_texctn[n_traj]
        key, subkey = jax.random.split(key)
        train_idxs_k, val_idxs_k = train_val_split_idxs(n_traj, t_exctn, val_size, subkey)
        train_idxs.append(train_idxs_k)
        val_idxs.append(val_idxs_k)
    return train_idxs, val_idxs


def get_traj_to_excitation_time_dict():
    keys1 = [x for x in range(1, 9)]
    values1 = [(3,13) for x in range(1, 9)]

    keys2 = [x for x in range(9, 16)]
    values2 = [(3,17) for x in range(9, 16)]

    keys3 = [x for x in range(16, 20)]
    values3 = [(3,22) for x in range(16, 20)]
    return dict(zip(keys1 + keys2 + keys3, values1 + values2 + values3))    


def construct_NNencoder_inputs(trajs: list):
    trajs_encoder = []
    for traj in trajs:
        # Create position features
        p_be = traj.p_e - traj.p_b
        pos_features = p_be

        # Create Euler angle features
        orient_features = compute_sin_cos_features(traj.phi_b)
        
        # Create linear velocity features
        lin_vel_features = jnp.hstack((traj.dp_b, traj.dp_e))
        
        # Create angular velocity features
        ang_vel_features = traj.dphi_b

        # Stack all features
        h = jnp.hstack((pos_features, orient_features, 
                        lin_vel_features, ang_vel_features))
        trajs_encoder.append(h)
    return trajs_encoder


def construct_PIencoder_inputs(trajs: list):
    trajs_encoder = []
    for traj in trajs:
        # Create position features
        pos_features = traj.p_e - traj.p_b

        # Create Euler angle features
        orient_features = compute_sin_cos_features(traj.phi_b)
        
        # Create linear velocity features
        lin_vel_features = traj.dp_e - traj.dp_b
        
        # Create angular velocity features
        ang_vel_features = jnp.hstack((
            jnp.multiply(orient_features[:,3:], traj.dphi_b),
            jnp.multiply(-orient_features[:,:3], traj.dphi_b)
        )) 

        # Stack all features
        h = jnp.hstack((pos_features, orient_features, 
                        lin_vel_features, ang_vel_features))
        trajs_encoder.append(h)
    return trajs_encoder


def construct_dyn_inputs(trajs):
    trajs_dyn = []
    for traj in trajs:
        # Orientation feature
        orient_features = compute_sin_cos_features(traj.phi_b)

        # Velocity features
        vel_features = jnp.hstack((traj.dp_b, traj.dphi_b))

        # Acceleration features
        acc_features = jnp.hstack((traj.ddp_b, traj.ddphi_b))
        h = jnp.hstack((orient_features, vel_features, acc_features))
        trajs_dyn.append(h)
    return trajs_dyn


def construct_FKdecoder_inputs(trajs):
    trajs_decoder = []
    for traj in trajs:
        # Pose features
        pose_features = jnp.hstack((traj.p_b, traj.phi_b))

        # Velocity features
        vel_features = jnp.hstack((traj.dp_b, traj.dphi_b))

        # Acceleration features
        h = jnp.hstack((pose_features, vel_features))
        trajs_decoder.append(h)
    return trajs_decoder


def construct_observations(trajs):
    trajs_y = []
    for traj in trajs:
        y = jnp.hstack((traj.p_e, traj.dp_e))
        trajs_y.append(y)
    return trajs_y


def compute_sin_cos_features(phi_b: jnp.ndarray):
    sin_phi_b = jnp.sin(phi_b)
    cos_phi_b = jnp.cos(phi_b)
    return jnp.hstack((sin_phi_b, cos_phi_b))


def fit_minmax_scalar(trajs: list, range: tuple = (-1,1)):
    all_trajs = jnp.vstack(trajs)
    scalar = JAXMinMaxScalar(feature_range=range)
    scalar.fit(all_trajs)
    return scalar


def scale_features(trajs: jnp.ndarray, scalar: JAXMinMaxScalar):
    return [scalar.transform(traj) for traj in trajs]


def divide_into_rolling_windows(trajs, window: int):
    rollouts = []
    for traj in trajs:
        traj_length = traj.shape[0]
        if traj_length > window:
            idxs = rolling_window(traj_length, window+1)
            rollouts.append(traj[idxs])
    return rollouts


def divide_into_sliding_windows(trajs, window: int):
    rollouts = []
    for traj in trajs:
        traj_length = traj.shape[0]
        idxs = sliding_window(traj_length, window+1)
        rollouts.append(traj[idxs])
    return rollouts


def divide_into_rolling_windows_and_stack(trajs, window: int):
    rollouts = divide_into_rolling_windows(trajs, window)
    return jnp.vstack(rollouts)


def divide_into_sliding_windows_and_stack(trajs, window: int):
    rollouts = divide_into_sliding_windows(trajs, window)
    return jnp.vstack(rollouts)


def rolling_window(traj_length:int, window: int):
    n_windows = traj_length - window + 1
    idxs = jnp.arange(n_windows)[:, None] + jnp.arange(window)[None, :]
    return idxs


def sliding_window(traj_length:int, window: int):
    n_windows = traj_length//(window - 1)
    idxs = jnp.arange(0, n_windows*(window - 1), window-1)[:,None] + jnp.arange(window)[None, :]
    return idxs    


def slice_trajs(trajs, idxs):
    s_trajs = [[traj.get_sliced_copy(slice) for slice in idxs_k] for traj, idxs_k in zip(trajs, idxs)]
    return list(itertools.chain.from_iterable(s_trajs))


def split_trajs_into_train_val(trajs, val_size: float, key: jax.random.PRNGKey):
    train_idxs, val_idxs = get_train_val_idxs(trajs, val_size, key)
    train_trajs = slice_trajs(trajs, train_idxs)
    val_trajs = slice_trajs(trajs, val_idxs)
    return train_trajs, val_trajs


def construct_train_val_datasets(train_trajs, val_trajs, rollout_length: int):
    # Encoder dataset
    enc_trajs_train = construct_PIencoder_inputs(train_trajs)
    enc_trajs_val = construct_PIencoder_inputs(val_trajs)
    U_enc_train = divide_into_rolling_windows_and_stack(enc_trajs_train, rollout_length)
    U_enc_val = divide_into_rolling_windows_and_stack(enc_trajs_val, rollout_length)

    # dyn dataset
    dyn_trajs_train = construct_dyn_inputs(train_trajs)
    dyn_traj_val = construct_dyn_inputs(val_trajs)

    dyn_scalar = fit_minmax_scalar(dyn_trajs_train)
    dyn_trajs_train_scaled = scale_features(dyn_trajs_train, dyn_scalar)
    dyn_traj_val_scaled = scale_features(dyn_traj_val, dyn_scalar)
    U_dyn_train = divide_into_rolling_windows_and_stack(dyn_trajs_train_scaled, rollout_length)[:,:-1,:]
    U_dyn_val = divide_into_rolling_windows_and_stack(dyn_traj_val_scaled, rollout_length)[:,:-1,:]

    # Decoder dataset
    dec_trajs_train = construct_FKdecoder_inputs(train_trajs)
    dec_trajs_val = construct_FKdecoder_inputs(val_trajs)
    U_dec_train = divide_into_rolling_windows_and_stack(dec_trajs_train, rollout_length)
    U_dec_val = divide_into_rolling_windows_and_stack(dec_trajs_val, rollout_length)

    # Outputs
    obs_trajs_train = construct_observations(train_trajs)
    obs_trajs_val = construct_observations(val_trajs)

    obs_scalar = fit_minmax_scalar(obs_trajs_train)
    obs_trajs_train_scaled = scale_features(obs_trajs_train, obs_scalar)
    obs_trajs_val_scaled = scale_features(obs_trajs_val, obs_scalar)
    Y_train = divide_into_rolling_windows_and_stack(obs_trajs_train_scaled, rollout_length)
    Y_val = divide_into_rolling_windows_and_stack(obs_trajs_val_scaled, rollout_length)

    # Construct dataset
    train_dataset = DLODataset(
        U_encoder=U_enc_train, 
        U_dyn=U_dyn_train, 
        U_decoder=U_dec_train, 
        Y=Y_train,
        encoder_scalar=None, 
        dyn_scalar=dyn_scalar, 
        output_scalar=obs_scalar
    )

    val_dataset = DLODataset(
        U_encoder=U_enc_val, 
        U_dyn=U_dyn_val, 
        U_decoder=U_dec_val, 
        Y=Y_val,
        encoder_scalar=None, 
        dyn_scalar=dyn_scalar, 
        output_scalar=obs_scalar
    )
    return train_dataset, val_dataset


def construct_test_dataset_from_trajs(
        test_trajs, 
        rollout_length: int, 
        scalars: DLODatasetScalars, 
        window_type: str,
        *, scale_outputs: bool
    ):
    window_divider = {
        'rolling':divide_into_rolling_windows_and_stack,
        'sliding': divide_into_sliding_windows_and_stack
    }
    divider = window_divider[window_type]

    # Encoder dataset
    enc_trajs = construct_PIencoder_inputs(test_trajs)
    U_enc = divider(enc_trajs, rollout_length)

    # RNN dataset
    dyn_trajs = construct_dyn_inputs(test_trajs)
    dyn_trajs_scaled = scale_features(dyn_trajs, scalars.dyn_scalar)
    U_dyn = divider(dyn_trajs_scaled, rollout_length)[:,:-1,:]

    # Decoder dataset
    dec_trajs = construct_FKdecoder_inputs(test_trajs)
    U_dec = divider(dec_trajs, rollout_length)

    # Outputs
    obs_trajs = construct_observations(test_trajs)
    if scale_outputs:
        obs_trajs_scaled = scale_features(obs_trajs, scalars.output_scalar)
    else:
        scalars.output_scalar = None
        obs_trajs_scaled = obs_trajs
    Y = divider(obs_trajs_scaled, rollout_length)

    # Construct dataset
    dataset = DLODataset(
        U_encoder=U_enc, 
        U_dyn=U_dyn, 
        U_decoder=U_dec, 
        Y=Y,
        encoder_scalar=scalars.encoder_scalar, 
        dyn_scalar=scalars.dyn_scalar, 
        output_scalar=scalars.output_scalar
    )
    return dataset


def main():
    pass
    

if __name__ == "__main__":
    main()