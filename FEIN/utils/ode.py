from enum import Enum, auto
from collections import namedtuple
from typing import Callable, Tuple
import jax.numpy as jnp
from jax.experimental.ode import odeint
from jax import lax


class IntegrationMethod(Enum):
    RK45 = auto()


IntegratorSetting = namedtuple(
    'IntegratorSetting', 
    ['dt', 'rtol', 'atol', 'method']
)


def simulate_ode(
    ode: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    x0: jnp.ndarray, 
    u: jnp.ndarray, 
    integrator_setting: IntegratorSetting
):
    """ Simulates an ODE starting from an initial
    state x0 and applies inputs from u vector. 
    
    :param ode: ode of the considered system
    :param x0: [nx,] initial state
    :param u: [N_sim x nu] input sequence
    :param integrator_setting: integrator settings such
            as relative and absolute tolerances as well
            integration step size and method

    :return: state trajectory of the system
    """
    dt = integrator_setting.dt
    method = integrator_setting.method
    if method == IntegrationMethod.RK45:
        rtol = integrator_setting.rtol
        atol = integrator_setting.atol

        ode_ = lambda x, t, u: ode(x, u)

        def body_fcn(carry, input):
            x_next = odeint(ode_, carry, jnp.array([0, dt]), input, rtol=rtol, atol=atol)[-1]
            return x_next, x_next

        _, outputs = lax.scan(body_fcn, x0, u)

    else:
        raise NotImplementedError

    return jnp.vstack((x0, outputs))