import jax.numpy as jnp
import equinox as eqx


from FEIN.utils.nn import MLPParameters
from FEIN.rfem_kinematics import models
from FEIN.rfem_kinematics import algorithms


class NNDecoder(eqx.Module):
    n_seg: int
    g_to_ye: eqx.Module

    def __init__(self, n_seg: int, mlp_params: MLPParameters, *, key) -> None:
        self.n_seg = n_seg
        self.g_to_ye = eqx.nn.MLP(**mlp_params.__dict__, key=key)

    @eqx.filter_jit
    def __call__(self, x_rnn, u_dec):
        x = jnp.concatenate((x_rnn, u_dec))
        pe_dpe_hat = self.g_to_ye(x)
        return pe_dpe_hat


class FKDecoder(eqx.Module):
    rfem_params: models.RFEMParameters
    
    def __init__(self, rfem_params: models.RFEMParameters) -> None:
        self.rfem_params = rfem_params

    @eqx.filter_jit
    def __call__(self, x_rnn, u_dec):
        """ Given state of the system return its outputs
        x_rnn = [q_rfem, dq_frem]
        u_dec = [q_b, dq_b]
        """
        model = models.create_learnable_rfem_custom_model(self.rfem_params)

        q_rfem, dq_rfem = jnp.split(x_rnn, 2)
        q_b, dq_b = jnp.split(u_dec, 2)
        q = jnp.concatenate((q_b, q_rfem))[:, jnp.newaxis]
        dq = jnp.concatenate((dq_b, dq_rfem))[:, jnp.newaxis]

        p_e, dp_e = algorithms.compute_markers_positions_and_velocities(model, q, dq)

        y = jnp.hstack((p_e, dp_e)).ravel()
        return y


class TrainableFKDecoder(eqx.Module):
    rfem_params: models.RFEMParameters
    rod_length: float
    rfem_lengths_sqrt: jnp.ndarray
    
    def __init__(self, rfem_params: models.RFEMParameters) -> None:
        self.rod_length = float(sum(rfem_params.lengths))
        self.rfem_params = rfem_params
        self.rfem_lengths_sqrt = jnp.sqrt(rfem_params.lengths)

    @eqx.filter_jit
    def __call__(self, x_rnn, u_dec):
        """ Given state of the system return its outputs
        x_rnn = [q_rfem, dq_frem]
        u_dec = [q_b, dq_b]
        """
        # Get 
        rfem_length = self.rfem_lengths_sqrt**2
        L_ = sum(rfem_length)
        scaling_ = self.rod_length / L_
        self.rfem_params.lengths = rfem_length * scaling_

        # model = models.create_rfem_custom_model(self.rfem_params)
        model = models.create_learnable_rfem_custom_model(self.rfem_params)

        q_rfem, dq_rfem = jnp.split(x_rnn, 2)
        q_b, dq_b = jnp.split(u_dec, 2)
        q = jnp.concatenate((q_b, q_rfem))[:, jnp.newaxis]
        dq = jnp.concatenate((dq_b, dq_rfem))[:, jnp.newaxis]

        p_e, dp_e = algorithms.compute_markers_positions_and_velocities(model, q, dq)

        y = jnp.hstack((p_e, dp_e)).ravel()
        return y