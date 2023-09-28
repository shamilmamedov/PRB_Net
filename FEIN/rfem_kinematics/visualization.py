from panda3d_viewer import Viewer, ViewerConfig
from pinocchio.visualize import Panda3dVisualizer, MeshcatVisualizer
import time
import cv2
import os
import shutil
import numpy as np

PANDA3D_CONFIG = ViewerConfig()
PANDA3D_CONFIG.enable_antialiasing(True, multisamples=5)
PANDA3D_CONFIG.enable_shadow(False)
PANDA3D_CONFIG.show_axes(False)
PANDA3D_CONFIG.show_grid(True)
PANDA3D_CONFIG.show_floor(True)
PANDA3D_CONFIG.enable_spotlight(False)
PANDA3D_CONFIG.enable_hdr(True)
PANDA3D_CONFIG.set_window_size(2025, 1500)


def visualize_robot(
        q: np.ndarray, 
        dt: float, 
        n_replays: int, 
        model, 
        geom_model, 
        vis_model=None, 
        visualizer='panda3d', 
        video_name=None
):
    if vis_model is None:
        vis_model = geom_model

    if visualizer == 'panda3d':
        viz = Panda3dVisualizer(model, geom_model, vis_model)
        viewer = Viewer(config=PANDA3D_CONFIG)
        viewer.set_background_color(((255, 255, 255)))
        # viewer.reset_camera((4, 3, 3), look_at=(1,0,1))
        viewer.reset_camera((4., 5.5, 1.6), look_at=(1,-0.5,1.4))
        viz.initViewer(viewer=viewer)
        viz.loadViewerModel(group_name=f'{model.name}')

        # set rfe colors
        for k, geom in enumerate(viz.visual_model.geometryObjects):
            if 'rfe' in geom.name or 'ee_ref' in geom.name:
                rgba = vis_model.geometryObjects[k].meshColor
                rgba = (rgba[0], rgba[1], rgba[2], rgba[3])
                viz.viewer.set_material(viz.visual_group, geom.name, rgba)

    if visualizer == 'meshcat':
        viz = MeshcatVisualizer(model, geom_model, vis_model)
        viz.initViewer()
        viz.loadViewerModel()
    
    capture = False
    if video_name is not None:
        capture = True 
        # create a folder to store the images
        capture_dir = 'tmp'
        if not os.path.exists(capture_dir):
            os.makedirs(capture_dir)

    for k in range(n_replays):
        if visualizer == 'panda3d':
            nsteps = len(q)
            for i in range(nsteps):
                t0 = time.time()
                viz.display(q[i])
                if capture and k == 0 and i>0:
                    viz.viewer.save_screenshot(f'{capture_dir}/img_{i}.png')
                t1 = time.time()
                elapsed_time = t1 - t0
                if dt is not None and elapsed_time < dt:
                    time.sleep(dt - elapsed_time)
        if visualizer == 'meshcat':
            viz.play(q, dt)
        time.sleep(2)
    viz.viewer.stop()

    if capture:
        images = [img for img in os.listdir(capture_dir) if img.startswith("img_")]
        images = sorted(images, key=lambda x: int(x.split('_')[1].split('.')[0]))
        frame = cv2.imread(os.path.join(capture_dir, images[0]))
        height, width, layers = frame.shape

        output_video = f'{video_name}.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
        video = cv2.VideoWriter(output_video, fourcc, 25, (width, height))

        for image in images:
            img_path = os.path.join(capture_dir, image)
            frame = cv2.imread(img_path)
            video.write(frame)

        video.release()

        # remove directory called tmp
        shutil.rmtree(capture_dir)
