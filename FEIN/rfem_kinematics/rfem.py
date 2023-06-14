from typing import List
import numpy as np
from dataclasses import dataclass
import jax.numpy as jnp


@dataclass(frozen=True)
class DLOParameters:
    """ Parameters of a cylindrical deformabal linear object
    """
    length: float = 2.
    diameter_inner: float = None
    diameter_outer: float = 0.05
    density: float = 2710.
    youngs_modulus: float = 7.0e+10
    shear_modulus: float = 2.7e+10
    normal_damping_coef: float = None
    tangential_damping_coef: float = None


def compute_rfe_lengths(L: float, n_seg: int) -> List[float]:
    """
    :param L: link length
    :param n_seg: number of segments

    :return: list with rfe lenghts
    """
    return jnp.array([L/(2*n_seg)] + [L/n_seg]*(n_seg-1) + [L/(2*n_seg)])


def compute_sde_joint_placements(rfe_lengths: list, frame: str = 'base'):
    """ Computes passive spring-damper elemnts 

    :param L: link length
    :param n_seg: number of segments
    :param frame: frame at which to calculate the placement
                  base = wrt the beginning of the link
                  parent = wrt to the parent joint
    
    :return: a list with X-position of the placmenet, Y- 
             and Z- are assumed to be zero
    """
    assert frame in ['base', 'parent']

    if frame == 'parent':
        return jnp.concatenate((jnp.array([0.]), rfe_lengths[:-1]))
    else:
        jpk = 0.
        jpositions = [jpk]
        for lk in rfe_lengths[:-1]:
            jpk += lk
            jpositions.append(jpk)
        return jnp.array(jpositions)


def compute_marker_frames_parent_joints(rfe_lengths: List, p_markers: List):
    """ Computes parent joints for each marker frame

    :param L: link length
    :param p_markers: marker positions in the base frame
    :param n_seg: number of segments
    """
    jpositions = compute_sde_joint_placements(rfe_lengths, frame='base')
    jpositions = jnp.array(jpositions)

    mparents = []
    for pk in p_markers:
        delta = pk - jpositions
        parent = int(jnp.where(delta > 0, delta, jnp.inf).argmin())
        mparents.append(parent)

    # NOTE there is a base joint which is not considered while getting
    # parent joints
    return mparents


def compute_marker_frames_placements(rfe_lengths: List, mparents: List, p_markers: List):
    """ Computes the placements of marker frames wrt to parent joint

    :param L: link length
    :param p_markers: marker positions in the base frame
    :param n_seg: number of segments
    """
    jpositions = compute_sde_joint_placements(rfe_lengths, frame='base')
    mplacements = []
    for pos_mk, par_mk in zip(p_markers, mparents):
        mplacements.append(pos_mk - jpositions[par_mk])
    return jnp.array(mplacements)


def compute_rfe_inertial_parameters(n_seg: int, dlo_params: DLOParameters):
    """ Computes inertial parameters of each rfe of the DLO

    :return: a list of masses, center of masses and ineertia tensors
    """
    L = dlo_params.length
    d = dlo_params.diameter_inner
    D = dlo_params.diameter_outer
    rho = dlo_params.density

    rfe_lengths = compute_rfe_lengths(L, n_seg)

    rfe_inertial_params = [inertial_params_hollow_cylinder(d, D, x, rho) for x in rfe_lengths]
    rfe_m = [x[0] for x in rfe_inertial_params]
    rfe_rc = [x[1] for x in rfe_inertial_params]
    rfe_I = [x[2] for x in rfe_inertial_params]
    return rfe_m, rfe_rc, rfe_I


def compute_rfe_sde_parameters(n_seg: int, dlo_params: DLOParameters):
    L = dlo_params.length
    d = dlo_params.diameter_inner
    D = dlo_params.diameter_outer
    E = dlo_params.youngs_modulus
    G = dlo_params.shear_modulus
    nu_bar = dlo_params.tangential_damping_coef
    nu = dlo_params.normal_damping_coef
    
    delta_l = L/n_seg
    # We are also interested in bending stiffness, not torsion
    sde_k = [spring_params_hollow_cylinder(d, D, delta_l, E, G)[1:,:]]*n_seg
    sde_d = [spring_params_hollow_cylinder(d, D, delta_l, nu, nu_bar)[1:,:]]*n_seg
    return sde_k, sde_d


def inertial_params_hollow_cylinder(d: float, D: float, l: float, rho):
    """ Computes inertial parameters of a hollow cylinder
    NOTE in inertia tensor it is assumed that off-diagonal terms
    are negligible comapred to diagonal terms and, thus, neglected

    :param d: inner diameter
    :param D: outer diameter
    :param l: length
    :param rho: denosity

    :return: (m, rc, I) tuple containing mass, vector
             to the center of mass and inertia tensor 
    """
    # Mass
    V = jnp.pi/4*l*(D**2 - d**2)
    m = rho*V

    # Center of mass
    rc = jnp.array([[l/2, 0., 0.]]).T

    # Second moments of inertia
    Ixx = m/8*(D**2 + d**2)
    Iyy = m/48*(3*D**2 + 3*d**2 + 4*l**2)
    Izz = m/48*(3*D**2 + 3*d**2 + 4*l**2)

    return m, rc, jnp.diag(jnp.array([Ixx, Iyy, Izz]))


def spring_params_hollow_cylinder(d: float, D: float, l: float, E: float, G: float):
    """ Equivalent spring parameters of a hollow cylindtical beam

    :param d: inner diameter
    :param D: outer diameter
    :param l: length
    :param E: Young modulus
    :param G: Shear modulus

    :return: array of stiffness parameters
    """
    r1 = d/2
    r2 = D/2

    t_ = np.pi*(r2**4 - r1**4)
    Jxx = t_/2
    Jyy = t_/4
    Jzz = t_/4

    kx = G*Jxx/l
    ky = E*Jyy/l
    kz = E*Jzz/l

    return jnp.array([[kx, ky, kz]]).T