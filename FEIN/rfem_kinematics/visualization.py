from panda3d_viewer import Viewer, ViewerConfig
from pinocchio.visualize import Panda3dVisualizer, MeshcatVisualizer
import time

PANDA3D_CONFIG = ViewerConfig()
PANDA3D_CONFIG.enable_antialiasing(True, multisamples=4)
PANDA3D_CONFIG.enable_shadow(False)
PANDA3D_CONFIG.show_axes(False)
PANDA3D_CONFIG.show_grid(True)
PANDA3D_CONFIG.show_floor(False)
PANDA3D_CONFIG.enable_spotlight(False)
PANDA3D_CONFIG.enable_hdr(True)
PANDA3D_CONFIG.set_window_size(1125, 1000)


def visualize_robot(q, dt, n_replays, model, geom_model, vis_model=None, visualizer='panda3d'):
    if vis_model is None:
        vis_model = geom_model

    if visualizer == 'panda3d':
        viz = Panda3dVisualizer(model, geom_model, vis_model)
        viewer = Viewer(config=PANDA3D_CONFIG)
        viewer.set_background_color(((255, 255, 255)))
        viewer.reset_camera((4, 3, 3), look_at=(1,0,0))
        viz.initViewer(viewer=viewer)
        viz.loadViewerModel(group_name=f'{model.name}')
    if visualizer == 'meshcat':
        viz = MeshcatVisualizer(model, geom_model, vis_model)
        viz.initViewer()
        viz.loadViewerModel()
    
    
    for _ in range(n_replays):
        viz.display(q[0, :])
        time.sleep(2)
        if visualizer == 'panda3d':
            viz.play(q[1:, :], dt)
        if visualizer == 'meshcat':
            viz.play(q[1:, :], dt)
        time.sleep(1)
    viz.viewer.stop()

