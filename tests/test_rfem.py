import jax.numpy as jnp
import matplotlib.pyplot as plt


from FEIN.rfem_kinematics import rfem

# unit test for compute_rfe_lengths
def test_compute_rfe_lengths():
    L = 1.
    n_seg = 3
    rfe_lengths = rfem.compute_rfe_lengths(L, n_seg)
    assert len(rfe_lengths) == n_seg + 1
    assert rfe_lengths[0] == L/(2*n_seg)
    assert rfe_lengths[-1] == L/(2*n_seg)
    assert all([rfe_lengths[i] == L/n_seg for i in range(1, n_seg)])


# unit test for compute_sde_joint_placements
def test_compute_sde_joint_placements():
    L = 1.
    n_seg = 3
    rfe_lengths = rfem.compute_rfe_lengths(L, n_seg)
    jpositions = rfem.compute_sde_joint_placements(rfe_lengths, frame='base')
    assert len(jpositions) == n_seg + 1
    jnp.allclose(jpositions[0], jnp.zeros((3, 1)))
    jnp.allclose(jpositions[-1], jnp.array([[L-rfe_lengths[-1], 0., 0.]]).T)
    for i in range(1, n_seg):
        jnp.allclose(jpositions[i], jnp.array([[L/(2*n_seg) + L/n_seg*i, 0., 0.]]).T)


    jpositions = rfem.compute_sde_joint_placements(rfe_lengths, frame='parent')
    assert len(jpositions) == n_seg + 1
    


# unit test for compute_marker_frames_parent_joints
def test_compute_marker_frames_parent_joints(viz=False):
    L = 1.
    n_seg = 3
    rfe_lengths = rfem.compute_rfe_lengths(L, n_seg)
    p_markers = [jnp.array([[0.1, 0., 0.]]).T, 
                 jnp.array([[0.45, 0., 0.]]).T, 
                 jnp.array([[0.9, 0., 0.]]).T,
                 jnp.array([[0.9, 0.05, 0.]]).T]
    mparents = rfem.compute_marker_frames_parent_joints(rfe_lengths, p_markers)
    assert len(mparents) == len(p_markers)
    assert mparents[0] == 0
    assert mparents[1] == 1
    assert mparents[2] == 3
    assert mparents[3] == 3

    # plot links and joint positions and markers to visually check
    # first get joint positions and connect them with lines that 
    # represent links
    if viz:
        jpositions = rfem.compute_sde_joint_placements(rfe_lengths, frame='base')
        fig, ax = plt.subplots()
        for i, jpos in enumerate(jpositions):
            ax.plot(jpos[0], jpos[1], 'o')
            ax.annotate(f'j{i}', (jpos[0], jpos[1]))
        for i, p_mk in enumerate(p_markers):
            ax.plot(p_mk[0], p_mk[1], 'x')
            ax.annotate(f'm{i}', (p_mk[0], p_mk[1]))
            ax.plot([jpositions[mparents[i]][0], p_mk[0]], 
                    [jpositions[mparents[i]][1], p_mk[1]], 'k--')
        plt.show()


# unit test for compute_marker_frames_placements
def test_compute_marker_frames_placements():
    L = 1.
    n_seg = 3
    rfe_lengths = rfem.compute_rfe_lengths(L, n_seg)
    p_markers = [jnp.array([[0.1, 0., 0.]]).T, 
                 jnp.array([[0.45, 0., 0.]]).T, 
                 jnp.array([[0.9, 0., 0.]]).T,
                 jnp.array([[0.9, 0.05, 0.]]).T]
    jpositions = rfem.compute_sde_joint_placements(rfe_lengths, frame='base')
    mparents = rfem.compute_marker_frames_parent_joints(rfe_lengths, p_markers)
    mplacements = rfem.compute_marker_frames_placements(rfe_lengths, mparents, p_markers)
    assert len(mplacements) == len(p_markers)
    assert jnp.allclose(mplacements[0], p_markers[0] - jpositions[mparents[0]])
    assert jnp.allclose(mplacements[1], p_markers[1] - jpositions[mparents[1]])
    assert jnp.allclose(mplacements[2], p_markers[2] - jpositions[mparents[2]])
    assert jnp.allclose(mplacements[3], p_markers[3] - jpositions[mparents[3]])


if __name__ == '__main__':
    test_compute_rfe_lengths()
    test_compute_sde_joint_placements()
    test_compute_marker_frames_parent_joints()
    test_compute_marker_frames_placements()