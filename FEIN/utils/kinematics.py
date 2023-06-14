import jax 
import jax.numpy as jnp
from enum import Enum, auto
from typing import Tuple


class JointType(Enum):
    U_ZY = auto()
    P_X = auto()
    P_Y = auto()
    P_Z = auto()
    P_XYZ = auto()
    FREE = auto()


JOINTTYPE_TO_NQ = {
    JointType.U_ZY: 2,
    JointType.P_X: 1,
    JointType.P_Y: 1,
    JointType.P_Z: 1,
    JointType.P_XYZ: 3,
    JointType.FREE: 6
}


@jax.jit
def rotz(x):
    sx = jnp.sin(x)
    cx = jnp.cos(x)
    return jnp.array([[cx, -sx, 0.],
                     [sx, cx, 0.],
                     [0., 0., 1.]])


@jax.jit
def roty(x):
    sx = jnp.sin(x)
    cx = jnp.cos(x)
    return jnp.array([[cx, 0., sx],
                     [0., 1., 0.],
                     [-sx, 0., cx]])

@jax.jit
def rotx(x):
    sx = jnp.sin(x)
    cx = jnp.cos(x)
    return jnp.array([[1., 0., 0.],
                      [0., cx, -sx],
                      [0., sx, cx]])

@jax.jit
def VecToso3(omg):
    """Converts a 3-vector to an so(3) representation

    :param omg: A 3-vector
    :return: The skew symmetric representation of omg
    """
    return jnp.array([[0.,      -omg[2,0],  omg[1,0]],
                     [omg[2,0],       0., -omg[0,0]],
                     [-omg[1,0], omg[0,0],       0.]])


@jax.jit
def inertia_vec2mat(I_vec: jnp.ndarray):
    """ Converts inertia matrix components stored as
    vector into a matrix

    I_vec = [Ixx Iyy Izz Ixy Ixz Iyz] = [6x1]

    I = [Ixx Ixy Ixz
         Ixy Iyy Iyz
         Ixz Iyz Izz]
    """
    return jnp.array([[I_vec[0,0], I_vec[3,0], I_vec[4,0]],
                      [I_vec[3,0], I_vec[1,0], I_vec[5,0]],
                      [I_vec[4,0], I_vec[5,0], I_vec[2,0]]])


def inertia_mat2vec(I: jnp.ndarray):
    pass


@jax.jit
def inertia_at_joint(R_ab, p_ba, m, I_b):
    """
    Compute the inertia in a body-fixed joint frame 'a'
    while inertia is defined in a body-fixed frame 'b'.
    (R_ab, p_ba) expresses the frame 'b' in the frame 'a'
    See "kim2012lie", Page 5
    :param m: body mass
    :param I: 3x3 inertia matrix
    :param R_ab: Rotation matrix
    :param p_ba: origin of 'b' w.r.t. 'a'
    :return: Spatial inertia matrix in 'a'
    """
    p = VecToso3(p_ba)
    return jnp.r_[jnp.c_[R_ab @ I_b @ R_ab.T + m * p.T @ p, m * p],
                  jnp.c_[m * p.T, m * jnp.eye(3)]]


@jax.jit
def Rp2Trans(R, p):
    """Converts a rotation matrix and a position vector into homogeneous
    transformation matrix
    :param R: A 3x3 rotation matrix
    :param p: A 3-vector
    :return: A homogeneous transformation matrix corresponding to the inputs
    """
    return jnp.r_[jnp.c_[R, p], [[0., 0., 0., 1.]]]


@jax.jit
def Trans2Rp(T):
    """Converts a homogeneous transformation matrix into a rotation matrix
    and position vector
    :param T: A homogeneous transformation matrix
    :return R: The corresponding rotation matrix,
    :return p: The corresponding position vector.
    """
    return T[0:3, 0:3], T[0:3, [3]]


@jax.jit
def TransInv(T):
    """Inverts a homogeneous transformation matrix

    :param T: A homogeneous transformation matrix
    :return: The inverse of T
    Uses the structure of transformation matrices to avoid taking a matrix
    inverse, for efficiency.
    """
    R, p = Trans2Rp(T)
    Rt = jnp.array(R).T
    return jnp.r_[jnp.c_[Rt, -jnp.dot(Rt, p)], [[0., 0., 0., 1.]]]


@jax.jit
def ad(V):
    """Calculate the 6x6 matrix [adV] of the given 6-vector

    :param V: A 6-vector spatial velocity
    :return: The corresponding 6x6 matrix [adV]

    Used to calculate the Lie bracket [V1, V2] = [adV1]V2
    """
    omgmat = VecToso3(V[:3,:])
    return jnp.r_[jnp.c_[omgmat, jnp.zeros((3, 3))],
                 jnp.c_[VecToso3(V[3:,:]), omgmat]]


@jax.jit
def Adjoint(T):
    """Computes the adjoint representation of a homogeneous transformation
    matrix

    :param T: A homogeneous transformation matrix
    :return: The 6x6 adjoint representation [AdT] of T

    """
    R, p = Trans2Rp(T)
    return jnp.r_[jnp.c_[R, jnp.zeros((3, 3))],
                 jnp.c_[jnp.dot(VecToso3(p), R), R]]


def jcalc_jax(type: str, q: jnp.ndarray, dq: jnp.ndarray = None):
    jtype_jcalc = {
        JointType.U_ZY: universal_ZY_joint_jax,
        JointType.P_XYZ: P_XYZ_joint_jax,
        JointType.FREE: free_joint_jax,
    }
    if type in jtype_jcalc:
        return jtype_jcalc[type](q, dq)
    else:
        raise NotImplementedError


@jax.jit
def universal_ZY_joint_jax(
    q: jnp.ndarray, 
    dq: jnp.ndarray = None
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """ Perform basic joint computations

    :param q: joint configuration
    :param dq: joint velocities [optional]

    :return: Homogenous transformation matrix R, 
             joint mapping matrix (joint Jacobian) S 
             and its derivatuive
    """
    # Rotatation matrix
    R = rotz(q[0,0]) @ roty(q[1,0])
    p = jnp.zeros((3,1))
    T = Rp2Trans(R, p)

    # Joint mapping matrix
    S = jnp.array([[-jnp.sin(q[1,0]), 0.], 
                   [0., 1.], 
                   [jnp.cos(q[1,0]), 0.],
                   [0., 0.],
                   [0., 0.],
                   [0., 0.]]
    )
                   
    # You don't always need velocity related term
    if dq is None:
        dS = None
    else:
        dS = jnp.array(([[-jnp.cos(q[1,0])*dq[1,0], 0.], 
                         [0., 0.], 
                         [-jnp.sin(q[1,0])*dq[1,0], 0.],
                         [0., 0.],
                         [0., 0.],
                         [0., 0.]])
        )
    return T, S, dS


@jax.jit
def P_XYZ_joint_jax(
    q: jnp.ndarray,
    dq: jnp.ndarray = None
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    :param q: joint configuration
    :param dq: joint velocities [optional]

    :return: Homogenous transformation matrix R, 
             joint mapping matrix (joint Jacobian) S 
             and its derivatuive
    """
    assert q.shape == (3,1)

    R = jnp.eye(3)
    T = Rp2Trans(R, q)

    S = jnp.vstack((jnp.zeros((3,3)), jnp.eye(3)))
    if dq is None:
        dS = None
    else:
        dS = jnp.zeros((6,3))
    return T, S, dS


@jax.jit
def free_joint_jax(
        q: jnp.ndarray,
        dq: jnp.ndarray = None
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    :param q: joint configuration (expressed in parent frame)
    :param dq: joint velocities (expressed in body frame)

    We chose the Pinocchio convention for the free joint, that is
    q = [global_base_position, global_base_orientation]
    :return: Homogenous transformation matrix R,
             joint mapping matrix (joint Jacobian) S
             and its derivative
    """
    assert q.shape == (6,1)

    q1, q2, q3 = q[3,0], q[4,0], q[5,0]
    
    s1, c1 = jnp.sin(q1), jnp.cos(q1)
    s2, c2 = jnp.sin(q2), jnp.cos(q2)
    s3, c3 = jnp.sin(q3), jnp.cos(q3)
    R = jnp.array([[c1*c2, -s1*c3 + s2*s3*c1, s1*s3 + s2*c1*c3],
                   [s1*c2, s1*s2*s3 + c1*c3, s1*s2*c3 - s3*c1],
                   [-s2, s3*c2, c2*c3]])
    T = Rp2Trans(R, q[:3, :])

    S_rot = jnp.array([[-s2, 0., 1.], [s3*c2, c3, 0.], [c2*c3, -s3, 0.]])
    S = jnp.r_[jnp.c_[jnp.zeros((3,3)), S_rot],
               jnp.c_[R.T, jnp.zeros((3,3))]]


    if dq is None:
        dS = None
    else:
        #dS = jnp.zeros((6,6))
        dq1, dq2, dq3 = dq[3,0], dq[4,0], dq[5,0]
        dS_rot = jnp.array([
            [-c2*dq2, 0, 0], 
            [-s2*s3*dq2 + c2*c3*dq3, -s3*dq3, 0], 
            [-s2*c3*dq2 - s3*c2*dq3, -c3*dq3, 0]
        ])
        dS_trans = jnp.array([
            [-s1*c2*dq1 - s2*c1*dq2, -s1*s2*dq2 + c1*c2*dq1, -c2*dq2],
            [-s1*s2*s3*dq1 + s1*s3*dq3 + s2*c1*c3*dq3 + s3*c1*c2*dq2 - c1*c3*dq1, 
                s1*s2*c3*dq3 + s1*s3*c2*dq2 - s1*c3*dq1 + s2*s3*c1*dq1 - s3*c1*dq3, 
                -s2*s3*dq2 + c2*c3*dq3
            ],
            [-s1*s2*c3*dq1 + s1*c3*dq3 - s2*s3*c1*dq3 + s3*c1*dq1 + c1*c2*c3*dq2, 
                -s1*s2*s3*dq3 + s1*s3*dq1 + s1*c2*c3*dq2 + s2*c1*c3*dq1 - c1*c3*dq3,
                -s2*c3*dq2 - s3*c2*dq3
            ]
        ])

        dS = jnp.r_[jnp.c_[jnp.zeros((3,3)), dS_rot],
                    jnp.c_[dS_trans, jnp.zeros((3,3))]]
    return T, S, dS