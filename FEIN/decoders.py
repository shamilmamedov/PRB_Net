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
        model = models.create_rfem_custom_model(self.rfem_params)

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
    p_marker: jnp.ndarray
    
    def __init__(self, rfem_params: models.RFEMParameters) -> None:
        self.rod_length = float(sum(rfem_params.lengths))
        self.rfem_params = rfem_params
        self.rfem_lengths_sqrt = jnp.sqrt(rfem_params.lengths)
        self.p_marker = jnp.array(jnp.copy(rfem_params.marker_positions[0]))

    @eqx.filter_jit
    def __call__(self, x_rnn, u_dec):
        """ Given state of the system return its outputs
        x_rnn = [q_rfem, dq_frem]
        u_dec = [q_b, dq_b]
        """
        model = self._get_updated_model_description()
        q, dq = self._xu_to_qdq(x_rnn, u_dec)
        y = self._compute_observations(model, q, dq)
        return y
    
    def _get_updated_model_description(self):
        self._update_rfem_lengths()
        self._update_marker_position()
        return models.create_rfem_custom_model(self.rfem_params)

    def _update_rfem_params(self):
        self._update_rfem_lengths()
        self._update_marker_position()

    def _update_rfem_lengths(self):
        rfem_lengths = self.rfem_lengths_sqrt**2
        L = jnp.sum(rfem_lengths)
        scaling = self.rod_length / L

        self.rfem_params.lengths = scaling * rfem_lengths

    def _update_marker_position(self):
        self.rfem_params.set_marker_positions([self.p_marker])

    @eqx.filter_jit
    def _xu_to_qdq(self, x, u):
        q_rfem, dq_rfem = jnp.split(x, 2)
        q_b, dq_b = jnp.split(u, 2)

        q = jnp.concatenate((q_b, q_rfem))[:, jnp.newaxis]
        dq = jnp.concatenate((dq_b, dq_rfem))[:, jnp.newaxis]
        return q, dq
    
    @eqx.filter_jit
    def _compute_observations(self, model, q, dq):
        p_e, dp_e = algorithms.compute_markers_positions_and_velocities(model, q, dq)
        return jnp.hstack((p_e, dp_e)).ravel()