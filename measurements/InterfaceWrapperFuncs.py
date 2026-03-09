import numpy as np
import importlib
import matplotlib.pyplot as plt
from pathlib import Path
import sys


from datetime import datetime
from dataclasses import dataclass
import pytz
from enum import Enum

from utils import circle_points, camera_view, noise
from classes import Camera, Body, PlanetImage, Pose

from filters import ChristianRobinson 
from trajectory.Constants import RAD_TO_DEG, DEG_TO_RAD, ARCSEC_TO_RAD, RAD_TO_ARCSEC

from trajectory.orbit import OrbitPropagator
from trajectory.Constants import G, Luna
from utils import circle_points


from .state_to_edgepoints import state_to_edgepoints, rotation_matrix_from_euler_angles_deg

np.set_printoptions(precision=3)

# Fixed mapping between renderer-effective camera basis and CR/CW basis.
# This is a proper rotation: 180 deg about +Z camera axis.
R_CR_FROM_RENDER = np.diag([-1.0, -1.0, 1.0])
R_RENDER_FROM_CR = R_CR_FROM_RENDER.T





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
    
    return  f, mu, n_pixels
        
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
    op = OrbitPropagator(state0, tf, dt, [body])
    op.simulate()

    return op, body

#make 3d 
def plot_traj(states, body):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(states[:,0], states[:,1], states[:,2], label='Trajectory')
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
    max_range = np.array([states[:,0].max()-states[:,0].min(), states[:,1].max()-states[:,1].min(), states[:,2].max()-states[:,2].min()]).max() / 2.0
    mid_x = (states[:,0].max()+states[:,0].min()) * 0.5
    mid_y = (states[:,1].max()+states[:,1].min()) * 0.5
    mid_z = (states[:,2].max()+states[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_title('Spacecraft Trajectory around Celestial Body')
    plt.legend()
    plt.show()


def plot_trajs(states_noisy, states_perfect, body):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(states_noisy[:,0], states_noisy[:,1], states_noisy[:,2], label='estimated Trajectory', color='r')
    ax.plot(states_perfect[:,0], states_perfect[:,1], states_perfect[:,2], label='noisy Trajectory', color='g', linestyle='--')
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
    max_range = np.array([states_noisy[:,0].max()-states_noisy[:,0].min(), states_noisy[:,1].max()-states_noisy[:,1].min(), states_noisy[:,2].max()-states_noisy[:,2].min()]).max() / 2.0
    mid_x = (states_noisy[:,0].max()+states_noisy[:,0].min()) * 0.5
    mid_y = (states_noisy[:,1].max()+states_noisy[:,1].min()) * 0.5
    mid_z = (states_noisy[:,2].max()+states_noisy[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_title('Spacecraft Trajectory around Celestial Body')
    plt.legend()
    plt.show()




def get_images(states, t, body, cam_offset, camera, num_images=10, cam_offset_units="auto"):
    """
    Generate rendered images along a trajectory with camera offset.
    
    Args:
        op: OrbitPropagator with simulated trajectory
        body: Body dictionary with radius and other properties
        cam_offset: Camera offset Euler angles [x, y, z] in degrees.
                   These describe the camera boresight relative to the spacecraft body frame,
                   where the spacecraft Z-axis points toward the planet center.
        camera: Camera specification dict with K, width, height
        num_images: Number of images to generate along trajectory
        cam_offset_units: Unused legacy parameter
    
    Returns:
        images: List of rendered images
        edges: List of detected edge point arrays
        outs: List of StateToEdgePointsResult objects
        states: List of spacecraft states
    """
    
    images = []
    edges = []
    outs = []
    states_out = []
    times = []
    
    # Convert camera offset Euler angles to rotation matrix
    # This represents rotation from spacecraft body frame (Z toward planet) to camera frame
    offset_rotation_matrix = rotation_matrix_from_euler_angles_deg(
        x_deg=cam_offset[0],
        y_deg=cam_offset[1],
        z_deg=cam_offset[2],
    )
    
   
    for i in range(num_images):
        state = states[i]
        print(f"State at t={t[i]:.1f}s: {state}")
        out = state_to_edgepoints(
            spacecraft_position_relative_to_body_center=state[0:3],
            attitude_offset_from_center_pointing=offset_rotation_matrix,
            k_matrix=camera["K"],
            width=camera["width"],
            height=camera["height"],
            noise_level=0.00,
            body_center_world=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            body_radius=body["radius"],
            attitude_offset_convention="world_to_camera",  # Rotation from spacecraft body to camera frame
            light_direction_world=np.array([0.5, 0, 0], dtype=np.float64),
            color_tolerance=4.5,
            keep_largest_component_only=True,
            fill_internal_holes=True,
            remove_speckles=True,
        )

        images.append(out.render.image)
        edges.append(out.edges.edge_coordinates_xy)
        outs.append(out)
        states_out.append(state)
        times.append(t[i])
    return images, edges, outs, states_out, times

def get_tpc(out):
    """
    Compute planet-to-camera rotation matrix (TPC) from render output.
    
    TPC transforms vectors from planet frame to camera frame (in CR convention).
    Assumes planet frame is aligned with world frame (same orientation, different origin).
    
    The renderer and CR algorithm use different camera frame conventions:
    - Renderer: standard OpenGL-style camera frame
    - CR: 180° rotation about Z-axis from renderer frame
    
    Args:
        out: StateToEdgePointsResult containing camera information
    
    Returns:
        TPC: 3x3 rotation matrix transforming vectors from planet to CR camera frame
    """
    # Extract camera axes from render output
    forward_world = out.camera.forward_world  # Camera Z-axis in world frame
    up_world = out.camera.up_world            # Camera Y-axis in world frame
    # right_world = np.cross(up_world, forward_world)  # Camera X-axis
    right_world = np.cross(forward_world, up_world)  # Camera X-axis
    
    # Build world-to-camera rotation in renderer convention
    # (rows = camera axes in world frame)
    R_world_to_render = np.stack([right_world, up_world, forward_world], axis=0)
    
    # Transform from renderer frame to CR frame
    # R_CR_FROM_RENDER is a 180° rotation about Z-axis
    R_world_to_cr = R_CR_FROM_RENDER @ R_world_to_render
    
    # Since planet frame is aligned with world frame (just different origin),
    # planet-to-camera rotation = world-to-camera rotation
    tpc = R_world_to_cr
    
    return tpc



def run_step(edges, rho_p_true, T_p_c, offset=None, print_stats=False, cr_alg=ChristianRobinson(np.eye(3), 1, 1, 1)):
    """
    Run Christian-Robinson algorithm on edge points.
    
    Args:
        edges: Edge points in homogeneous coordinates [x, y, 1]
        rho_p_true: True vector from planet to camera in planet frame
        T_p_c: Planet-to-camera rotation matrix
        offset: Unused legacy parameter
        print_stats: Whether to print diagnostic statistics
        cr_alg: Christian-Robinson algorithm instance
    
    Returns:
        r_c_est: Estimated camera-to-planet vector in camera frame
        r_p_est: Estimated camera-to-planet vector in planet frame
    
    Note:
        rho: P->C (planet to camera)
        r: C->P (camera to planet)
        i.e. r = -rho
    """
    rho_c_true = T_p_c @ rho_p_true  # P->C in camera frame
    r_c_true = -rho_c_true  # C->P in camera frame
    r_p_true = -rho_p_true  # C->P in planet frame

    # Calculate each time there is a new image
    pose = Pose(rho_p_true, T_p_c)
    
    # Run algorithm (gives vector from camera TO planet in camera frame)
    r_c_est = cr_alg.run(edges, pose.T_p_c)
    
    # Transform estimate to planet frame
    r_p_est = pose.T_p_c.T @ r_c_est

    # Print statistics
    if print_stats:
        with np.printoptions(precision=2, suppress=True):
            print(f"Calc Position:\n - (C->P in camera frame): {r_c_est} ({np.linalg.norm(r_c_est):.2f} km)")
            print(f" - (C->P in planet frame): {r_p_est} ({np.linalg.norm(r_p_est):.2f} km)")
            print(f"Actual position:\n - (C->P in camera frame): {r_c_true} ({np.linalg.norm(r_c_true):.2f} km)")
            print(f" - (C->P in planet frame): {r_p_true} ({np.linalg.norm(r_p_true):.2f} km)")
            print(f"Est Error - Actual:\n - (Camera frame): {r_c_est - r_c_true} ({np.linalg.norm(r_c_est - r_c_true):.2f} km)")
            print(f" - (Planet frame): {r_p_est - r_p_true} ({np.linalg.norm(r_p_est - r_p_true):.2f} km)")
            
            dist_error = abs(np.linalg.norm(r_c_true) - np.linalg.norm(r_c_est))
            dist_error_pct = 100 * dist_error / np.linalg.norm(r_c_true)
            print(f"Distance error: {dist_error:.2f} km ({dist_error_pct:.6f}%)")
            
            # Avoid division by zero and numerical issues in arccos
            cos_angle = np.dot(r_c_est, r_c_true) / (np.linalg.norm(r_c_est) * np.linalg.norm(r_c_true))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angular_error_deg = np.degrees(np.arccos(cos_angle))
            print(f"Angular error: {angular_error_deg:.4f} degrees")
    
    return r_c_est, r_p_est



def plot_est_dist_xyz(
    times,
    r_c_trues,
    r_c_ests,
    time_limits=None,
    dist_limits=None,
    x_limits=None,
    y_limits=None,
    z_limits=None,
):
    """
    Plot estimated vs true distance and vector components over time.

    Args:
        times: Array of time points.
        r_c_trues: Array of true camera-to-planet vectors in camera frame.
        r_c_ests: Array of estimated camera-to-planet vectors in camera frame.
        time_limits: Optional (t_min, t_max) for x-axis on all subplots.
        dist_limits: Optional (min, max) for distance subplot y-axis.
        x_limits: Optional (min, max) for X component subplot y-axis.
        y_limits: Optional (min, max) for Y component subplot y-axis.
        z_limits: Optional (min, max) for Z component subplot y-axis.
    """
    times = np.asarray(times)
    r_c_trues = np.asarray(r_c_trues)
    r_c_ests = np.asarray(r_c_ests)

    r_c_true_dists = np.linalg.norm(r_c_trues, axis=1)
    r_c_est_dists = np.linalg.norm(r_c_ests, axis=1)

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    
    axes[0].plot(times, r_c_true_dists, label="True Distance (km)", marker="o")
    axes[0].plot(times, r_c_est_dists, label="Estimated Distance (km)", marker="x")
    axes[0].set_ylabel("Distance (km)")
    axes[0].set_title("Camera-to-Planet Distance")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(times, r_c_trues[:, 0], label="True X", marker="o")
    axes[1].plot(times, r_c_ests[:, 0], label="Estimated X", marker="x")
    axes[1].set_ylabel("X (km)")
    axes[1].set_title("Camera-Frame X Component")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(times, r_c_trues[:, 1], label="True Y", marker="o")
    axes[2].plot(times, r_c_ests[:, 1], label="Estimated Y", marker="x")
    axes[2].set_ylabel("Y (km)")
    axes[2].set_title("Camera-Frame Y Component")
    axes[2].legend()
    axes[2].grid(True)

    axes[3].plot(times, r_c_trues[:, 2], label="True Z", marker="o")
    axes[3].plot(times, r_c_ests[:, 2], label="Estimated Z", marker="x")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_ylabel("Z (km)")
    axes[3].set_title("Camera-Frame Z Component")
    axes[3].legend()
    axes[3].grid(True)

    if time_limits is not None:
        for ax in axes:
            ax.set_xlim(time_limits)
    if dist_limits is not None:
        axes[0].set_ylim(dist_limits)
    if x_limits is not None:
        axes[1].set_ylim(x_limits)
    if y_limits is not None:
        axes[2].set_ylim(y_limits)
    if z_limits is not None:
        axes[3].set_ylim(z_limits)

    plt.tight_layout()
    plt.show()


def plot_est_dist_xyz_errors(
    times,
    r_c_trues,
    r_c_ests,
):
    
    
    times = np.asarray(times)
    r_c_trues = np.asarray(r_c_trues)
    r_c_ests = np.asarray(r_c_ests)

    r_c_true_dists = np.linalg.norm(r_c_trues, axis=1)
    r_c_est_dists = np.linalg.norm(r_c_ests, axis=1)

    dist_error = r_c_est_dists - r_c_true_dists
    X_error = r_c_ests[:, 0] - r_c_trues[:, 0]
    Y_error = r_c_ests[:, 1] - r_c_trues[:, 1]
    Z_error = r_c_ests[:, 2] - r_c_trues[:, 2]

    fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    
    axes[0].plot(times, dist_error, label="Distance Error (km)", marker="o")    
    axes[0].set_ylabel("Distance error (km)")
    axes[0].set_title("Camera-to-Planet Distance Error")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(times, X_error, label="X Error (km)", marker="o")
    axes[1].plot(times, Y_error, label="Y Error (km)", marker="o")
    axes[1].plot(times, Z_error, label="Z Error (km)", marker="o")
    axes[1].set_ylabel("X,Y,Z error (km)")
    axes[1].set_title("Camera-Frame error Components")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

