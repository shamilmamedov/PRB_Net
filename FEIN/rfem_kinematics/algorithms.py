import jax
import jax.numpy as jnp
import equinox as eqx
from jax import lax


import FEIN.utils.kinematics as jutils
from FEIN.rfem_kinematics.models import RobotDescription


def fwd_kinematics(model: RobotDescription, q: jnp.ndarray):
    """ Computes forward kinematics of a robot described 
    by model

    :param model: model decription
    :param q: configuration of the robot

    :return: a tupe of joint transformation and frame transformations
    """
    # Get idxs of joints in q vector
    k = 0
    q_idxs = []
    for nq in model.jnqs:
        q_idxs.append(jnp.arange(k,k+nq))
        k += nq

    # Joint trasnformations
    o_T_j = jnp.zeros((len(q_idxs), 4, 4))
    for k, qk_idxs in enumerate(q_idxs):
        Tjk, _, _ = jutils.jcalc_jax(model.jtypes[k], q[qk_idxs])
        Tk = lax.dot(model.jplacements[k]['T'], Tjk)
        if model.jparents[k] != -1:
            o_T_j = o_T_j.at[k].set(
                lax.dot(o_T_j[model.jparents[k]], Tk)
            )
        else:
            o_T_j = o_T_j.at[k].set(Tk)

    # Frame transformations
    o_T_f = dict() # dictionary of frame transformations
    for k in range(model.n_frames):
        Tk = model.fplacements[k]['T']
        o_T_f[k] = jnp.dot(o_T_j[model.fparents[k]], Tk)
    
    return o_T_j, o_T_f


def fwd_velocity_kinematics(model: RobotDescription, q: jnp.ndarray, dq: jnp.ndarray):
    # Get idxs of joints in q vector
    k = 0
    q_idxs = []
    for nq in model.jnqs:
        q_idxs.append(jnp.arange(k,k+nq))
        k += nq

    Vj = jnp.zeros((len(q_idxs), 6, 1))
    o_T_j = jnp.zeros((len(q_idxs), 4, 4))
    Tj, S, dS = zip(*[jutils.jcalc_jax(jti, q[qi_idxs], dq[qi_idxs]) 
                      for jti, qi_idxs in zip(model.jtypes, q_idxs)])
    for i, qi_idxs in enumerate(q_idxs):
        if i == 0:
            V_λ = jnp.zeros((6,1))
        else:
            V_λ = Vj[i-1]

        # Describe i-frame in λ-frame (parent-frame of i)
        T_λj = model.jplacements[i]['T']
        T_λi = lax.dot(T_λj, Tj[i])

        # Describe i-th joint frame in 0-frame
        if model.jparents[i] != -1:
            o_T_j = o_T_j.at[i].set(
                lax.dot(o_T_j[model.jparents[i]], T_λi)
            )
        else:
            o_T_j = o_T_j.at[i].set(T_λi)

        Ad_T_iλ = jutils.Adjoint(jutils.TransInv(T_λi))

        # Velocity and acceleration of i-th body
        Vj = Vj.at[i].set(
            lax.dot(Ad_T_iλ, V_λ) + lax.dot(S[i], dq[qi_idxs])
        )

    Vf = dict()
    o_Vf = dict()
    for k in range(model.n_frames):
        V_parent_joint = Vj[model.fparents[k]]
        
        Tk = model.fplacements[k]['T']
        Vf[k] = lax.dot(jutils.Adjoint(jutils.TransInv(Tk)), V_parent_joint)

        o_T_f = lax.dot(o_T_j[model.fparents[k]], Tk)
        o_R_f = jutils.Trans2Rp(o_T_f)[0]

        o_Vf[k] = lax.dot(jax.scipy.linalg.block_diag(o_R_f, o_R_f), Vf[k])

    return Vj, o_Vf


def fwd_joint_position_and_velocity_kinematics(
        model: RobotDescription, 
        q: jnp.ndarray, 
        dq: jnp.ndarray
):
    # Get idxs of joints in q vector
    k = 0
    q_idxs = []
    for nq in model.jnqs:
        q_idxs.append(jnp.arange(k,k+nq))
        k += nq

    Vj = jnp.zeros((len(q_idxs), 6, 1))
    o_T_j = jnp.zeros((len(q_idxs), 4, 4)) 
    Tj, S, dS = zip(*[jutils.jcalc_jax(jti, q[qi_idxs], dq[qi_idxs]) 
                      for jti, qi_idxs in zip(model.jtypes, q_idxs)])
    for i, qi_idxs in enumerate(q_idxs):
        if i == 0:
            V_λ = jnp.zeros((6,1))
        else:
            V_λ = Vj[i-1]

        # Describe i-frame in λ-frame (parent-frame of i)
        T_λj = model.jplacements[i]['T']
        T_λi = lax.dot(T_λj, Tj[i])

        # Describe i-th joint frame in 0-frame
        if model.jparents[i] != -1:
            o_T_j = o_T_j.at[i].set(
                lax.dot(o_T_j[model.jparents[i]], T_λi)
            )
        else:
            o_T_j = o_T_j.at[i].set(T_λi)

        Ad_T_iλ = jutils.Adjoint(jutils.TransInv(T_λi))

        # Velocity and acceleration of i-th body
        Vj = Vj.at[i].set(
            lax.dot(Ad_T_iλ, V_λ) + lax.dot(S[i], dq[qi_idxs])
        )
    return o_T_j, Vj


def compute_markers_positions_and_velocities(model, q, dq):
    o_T_j, Vj = fwd_joint_position_and_velocity_kinematics(model, q, dq)

    p_markers = jnp.zeros((model.n_frames, 3, 1))
    for k in range(model.n_frames):
        Tk = model.fplacements[k]['T']
        o_T_f = jnp.dot(o_T_j[model.fparents[k]], Tk)
        pm_k = jutils.Trans2Rp(o_T_f)[1]
        p_markers = p_markers.at[k].set(pm_k)

    dp_markers = jnp.zeros((model.n_frames, 3, 1))
    for k in range(model.n_frames):
        V_parent_joint = Vj[model.fparents[k]]
        
        Tk = model.fplacements[k]['T']
        Vf = lax.dot(jutils.Adjoint(jutils.TransInv(Tk)), V_parent_joint)

        o_T_f = jnp.dot(o_T_j[model.fparents[k]], Tk)
        o_R_f = jutils.Trans2Rp(o_T_f)[0]

        o_Vf = lax.dot(jax.scipy.linalg.block_diag(o_R_f, o_R_f), Vf)

        dp_markers = dp_markers.at[k].set(o_Vf[3:,:])
    return p_markers.squeeze(), dp_markers.squeeze()


def compute_markers_positions(model, q):
    """ Computes markers positions for a given configuration
    marker positions are organized as follows
    [p1_x, p1_y, p1_x, p2_x, p2_y, p2_z, ...., pn_x, pn_y, pn_z] 

    :param model: model description
    :param q: configuration of the robot

    :return: jax array with markets positions
    """
    o_T_j, o_T_f = fwd_kinematics(model, q)
    p_markers = []
    for key, value in o_T_f.items():
        p_markers.append(jutils.Trans2Rp(value)[1].T)
    return jnp.vstack(p_markers)

vcompute_markers_positions = eqx.filter_vmap(compute_markers_positions, in_axes=(None, 0))       


def compute_markers_velocities(model, q, dq):
    """Computes linear velocities of the markers"""
    Vj, o_Vf = fwd_velocity_kinematics(model, q, dq)
    dp_markers = []
    for key, value in o_Vf.items():
        dp_markers.append(value[3:,:].T)
    return jnp.vstack(dp_markers)


if __name__ == "__main__":
    pass
