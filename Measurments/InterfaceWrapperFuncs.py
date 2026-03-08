import numpy as np
import importlib
import matplotlib.pyplot as plt
from pathlib import Path
import sys
parent_dir = Path.cwd().parent.resolve()
sys.path.append(str(parent_dir))

np.set_printoptions(precision=3)
import state_to_edgepoints as state_to_edgepoints_module

state_to_edgepoints_module = importlib.reload(state_to_edgepoints_module)
state_to_edgepoints = state_to_edgepoints_module.state_to_edgepoints

import sys



from datetime import datetime
from dataclasses import dataclass
import pytz
from enum import Enum

from utils import circle_points, camera_view, noise
from classes import Camera, Body, PlanetImage, Pose

from filters import ChristianRobinson 
from Constants import RAD_TO_DEG, DEG_TO_RAD, ARCSEC_TO_RAD, RAD_TO_ARCSEC

import trajectory as T
from PlanetaryData import Luna
from Constants import G
from utils import circle_points





class CameraModel(Enum):
        SAGITTA = 0
        ST1 = 1


def Set_camera():
    
    camera_model = CameraModel.SAGITTA

    match camera_model:
        case CameraModel.SAGITTA:
            # Camera specs 
            # e.g. https://satsearch.co/products/arcsec-sagitta-star-tracker

            # Inputs
            fov = 25.2 * DEG_TO_RAD
            f = 2500e-3               # focal length
            n_pixels = np.array([2048, 2048])   # Assumed from previous star tracker

            cross_boresight_sigma = 2 * ARCSEC_TO_RAD
            boresight_sigma = 10 * ARCSEC_TO_RAD 
            # Calculated
            mu_angle = fov/n_pixels[0]       # pixel angle resolution [rad/px]
            mu = f * mu_angle             # pixel spacing [m/px]

        case CameraModel.ST1:
            # Camera specs 
            # e.g. https://spinworks.pt/products/star-tracker-st1/

            # Inputs
            fov_diag = 18.2 * DEG_TO_RAD
            f = 50e-3               # focal length
            n_pixels = np.array([2048, 2048])
            mu = 5.5e-6             # pixel spacing [m]
            cross_boresight_sigma = 2 * ARCSEC_TO_RAD / 3 # Gives in 3-sigma

            # Calculated
            mu_angle = mu / f       # pixel angle resolution [rad]
            fov = fov_diag / np.sqrt(2)
    
    return mu, f, n_pixels
        
def gen_traj(Starting_state=None, dt=0.1, tf_orbital_period_fraction=0.5):
    body = Luna # could be moon
    body["position"] = [0,0,0]

    # State
    if Starting_state == None:
        r = 2500 + body["radius"]
        v = np.sqrt(body["mu"] / r) + 0.5 # km/s, slightly above circular velocity
        state0 = [
            r, 0, 500,
            0, v, 0]
    else :
        state0 = Starting_state
    # Time
    

    # Simulating Half an orbit
    r = np.linalg.norm(state0[0:3])
    orb_period = 2 * np.pi * np.sqrt(r*r*r/body["mu"])
    print(f"Orbital period (T): {orb_period}s")
    tf = tf_orbital_period_fraction * orb_period

    # Propagator
    op = T.OrbitPropagator(state0, tf, dt, [body])
    op.simulate()

    return op, body

#make 3d 
def plot_traj(op, body):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(op.states[:,0], op.states[:,1], op.states[:,2], label='Trajectory')
    # Plot the celestial body as a sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = body["radius"] * np.cos(u) * np.sin(v)
    y = body["radius"] * np.sin(u) * np.sin(v)
    z = body["radius"] * np.cos(v)
    ax.plot_surface(x, y, z, color='b', alpha=0.5)
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    # scale all axis equally
    max_range = np.array([op.states[:,0].max()-op.states[:,0].min(), op.states[:,1].max()-op.states[:,1].min(), op.states[:,2].max()-op.states[:,2].min()]).max() / 2.0
    mid_x = (op.states[:,0].max()+op.states[:,0].min()) * 0.5
    mid_y = (op.states[:,1].max()+op.states[:,1].min()) * 0.5
    mid_z = (op.states[:,2].max()+op.states[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_title('Spacecraft Trajectory around Celestial Body')
    plt.legend()
    plt.show()



def get_images(op, body, camera, num_images=10):

    indices = np.linspace(0, len(op.states[:-1])-1, num_images, dtype=int)
    images = []
    edges = []
    outs = []
    states = []
    

    for i in indices:
        state = op.states[i]
        print(f"State at t={op.ts[i]:.1f}s: {state}")
        out = state_to_edgepoints(
            spacecraft_position_relative_to_body_center=state[0:3],
            attitude_offset_from_center_pointing=np.eye(3),
            k_matrix= camera["K"],
            width=camera["width"],
            height=camera["height"],
            noise_level=0.00,
            body_center_world=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            body_radius=body["radius"],
            attitude_offset_convention="world_to_camera",
            light_direction_world=np.array([0.5, 0, 0], dtype=np.float64),
            color_tolerance=4.5,
            keep_largest_component_only=True,
            fill_internal_holes=True,
            remove_speckles=True,
        )

        images.append(out.render.image)
        edges.append(out.edges.edge_coordinates_xy)
        outs.append(out)
        states.append(state)
    return images, edges, outs, states


def run_step(edges, rho_p_true, T_p_c, offset = None , print_stats = False,  cr_alg = ChristianRobinson(np.eye(3), 1, 1, 1)):
    """
    rho: P->C
    
    r: C->P
    
    i.e. 
    r = -rho
    """
    rho_c_true = T_p_c @ rho_p_true # P->C in camera frame (i.e. as seen looking INTO the lens from the planet)
    r_c_true = -rho_c_true
    r_p_true = -rho_p_true

    #if r_c_true[2] < 0:
    #    raise ValueError("Planet is behind the camera")

    # Calculate each time there is a new image
    pose = Pose(rho_p_true, T_p_c)
    

    if offset is not None:
        offset = np.asarray(offset)
    

    # Run algorithm (gives vector from camera TO planet in camera frame)
    r_c_est = cr_alg.run(edges, pose.T_p_c)
    
    # Vector directions are flipped
    r_p_est = pose.T_p_c.T @ r_c_est


    # Print statistics
    if print_stats:
        with np.printoptions(precision=2, suppress=True):
            print(f"Calc Position:\n - (C->P in camera frame): {r_c_est} ({np.linalg.norm(r_c_est)})m")
            print(f" - (C->P in planet frame): {r_p_est} ({np.linalg.norm(r_p_est)})m")
            print(f"Actual position:\n - (C->P in camera frame): {r_c_true} ({np.linalg.norm(r_c_true)})m")
            print(f" - (C->P in planet frame): {r_p_true} ({np.linalg.norm(r_p_true)})m")
            print(f"Est Error - Actual:\n - (Camera frame): {r_c_est - r_c_true} ({np.linalg.norm(r_c_est - r_c_true)})m")
            print(f" - (Planet frame): {r_p_est - r_p_true} ({np.linalg.norm(r_p_est - r_p_true)})m")
            
            print(f"Distance error: {abs(np.linalg.norm(r_c_true) - np.linalg.norm(r_c_est))} m ({100*abs(np.linalg.norm(r_c_true) - np.linalg.norm(r_c_est))/np.linalg.norm(r_c_true):.10f}%)")
            
            print(f"Angular error: {np.degrees(np.arccos(np.dot(r_c_est, r_c_true) /(np.linalg.norm(r_c_est)*np.linalg.norm(r_c_true)))):.2f} degrees")
    return r_c_est, r_p_est