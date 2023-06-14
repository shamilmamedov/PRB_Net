from sklearn.preprocessing import MinMaxScaler
import numpy as np
import jax.numpy as jnp

import training.preprocess_data as data_prpr


def test_minmax_scalar():
    for n_trajs in [[1], [17], [18], [19]]:
        trajs = data_prpr.load_trajs(n_trajs)

        U_enc = data_prpr.construct_NNencoder_inputs(trajs)

        # Test fit method
        own_scalar = data_prpr.fit_minmax_scalar(U_enc, range=(-1,1))
        sk_scalar = MinMaxScaler(feature_range=(-1,1)).fit(U_enc[0])

        np.testing.assert_almost_equal(sk_scalar.data_max_, np.asarray(own_scalar.data_max_))
        np.testing.assert_almost_equal(sk_scalar.data_min_, np.asarray(own_scalar.data_min_))

        # Test transform method
        U_enc_scaled_own = own_scalar.transform(U_enc[0])
        U_enc_scaled_sk = sk_scalar.transform(U_enc[0])

        np.testing.assert_almost_equal(U_enc_scaled_sk, np.asarray(U_enc_scaled_own), decimal=2)


def test_minmax_vtransform():
    n_trajs = [19]
    trajs = data_prpr.load_trajs(n_trajs)
    enc_trajs = data_prpr.construct_NNencoder_inputs(trajs)
    scalar = data_prpr.fit_minmax_scalar(enc_trajs, range=(-1,1))

    rollout_length = 250
    enc_trajs_scaled = data_prpr.scale_features(enc_trajs, scalar)
    U_enc_1 = data_prpr.divide_into_rolling_windows_and_stack(enc_trajs_scaled, rollout_length)

    U_enc_2 = data_prpr.divide_into_rolling_windows_and_stack(enc_trajs, rollout_length)
    U_enc_2 = scalar.vtransform(U_enc_2)

    jnp.allclose(U_enc_1, U_enc_2)
