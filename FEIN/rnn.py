import equinox as eqx
from dataclasses import dataclass
import jax.numpy as jnp
from jax import lax


@dataclass(frozen=True)
class RNNParameters:
    input_size: int = 6
    hidden_size: int = 18


class RNN(eqx.Module):
    state_transition: eqx.Module

    def __init__(self, rnn_params: RNNParameters, *, key) -> None:
        self.state_transition = eqx.nn.GRUCell(**rnn_params.__dict__, key=key)

    @eqx.filter_jit
    def __call__(self, x_0, U):
        def body_fcn(carry, input):
            x_next = self.state_transition(input, carry)
            return x_next, x_next

        _, outputs = lax.scan(body_fcn, x_0, U)
        return jnp.vstack((x_0, outputs))