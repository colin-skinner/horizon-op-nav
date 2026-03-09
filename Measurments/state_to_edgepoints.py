from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional
import matplotlib.pyplot as plt

import numpy as np

from Image_2_points import EdgeDetectionResult, find_body_edge_pixels
from SphereGenerator import (
	CameraSettings,
	CelestialBody,
	NoiseSettings,
	RenderResult,
	render_celestial_body,
)


@dataclass(frozen=True)
class StateToEdgePointsResult:
	render: RenderResult
	edges: EdgeDetectionResult
	camera: CameraSettings


def _validate_rotation_matrix(rotation_matrix: np.ndarray) -> np.ndarray:
	r = np.asarray(rotation_matrix, dtype=np.float64)
	if r.shape != (3, 3):
		raise ValueError("camera_rotation_matrix must have shape (3, 3).")

	orthogonality_error = np.linalg.norm((r @ r.T) - np.eye(3))
	det = float(np.linalg.det(r))
	if orthogonality_error > 1e-6 or abs(det - 1.0) > 1e-6:
		raise ValueError(
			"camera_rotation_matrix must be a proper rotation matrix "
			"(orthonormal with determinant +1)."
		)
	return r

def plot_edge_points(out, w, h, x_limits=None, y_limits=None):
	fig, ax = plt.subplots(figsize=(10, 5.6), dpi=120)
	ax.imshow(out.render.image)
	edge_xy = out.edges.edge_coordinates_xy
	ax.scatter(edge_xy[:, 0], edge_xy[:, 1], s=0.9, c="yellow", alpha=0.8)
	ax.set_title("state_to_edgepoints wrapper test (relative state + relative attitude) [km]")
	if x_limits is None:
		ax.set_xlim(0, w - 1)
	else:
		ax.set_xlim(x_limits)
	if y_limits is None:
		ax.set_ylim(h - 1, 0)
	else:
		ax.set_ylim(y_limits)
	ax.set_xlabel("x [px]")
	ax.set_ylabel("y [px]")
	plt.tight_layout()
	plt.show()


def rotation_matrix_from_euler_angles_deg(x_deg: float = 0.0, y_deg: float = 0.0, z_deg: float = 0.0) -> np.ndarray:
	"""Build a 3x3 rotation matrix from Euler angles in degrees.

	The returned matrix uses the composition R = Rz @ Ry @ Rx, where
	Rx, Ry, Rz are rotations about x, y, z axes respectively.
	
	When used with attitude_offset_from_center_pointing:
	- The spacecraft body frame has Z-axis pointing toward the planet center.
	- These Euler angles describe the camera boresight relative to that frame.
	- Example: [0, 10, 0] means camera boresight is 10° rotated about Y-axis
	  from the spacecraft Z-axis (which points at planet).
	
	Args:
	    x_deg: Rotation about X-axis in degrees
	    y_deg: Rotation about Y-axis in degrees
	    z_deg: Rotation about Z-axis in degrees
	
	Returns:
	    3x3 rotation matrix representing spacecraft-to-camera transformation
	"""
	rx = np.deg2rad(float(x_deg))
	ry_ = np.deg2rad(float(y_deg))
	rz = np.deg2rad(float(z_deg))

	cx, sx = np.cos(rx), np.sin(rx)
	cy, sy = np.cos(ry_), np.sin(ry_)
	cz, sz = np.cos(rz), np.sin(rz)

	rx_m = np.array(
		[
			[1.0, 0.0, 0.0],
			[0.0, cx, -sx],
			[0.0, sx, cx],
		],
		dtype=np.float64,
	)
	ry_m = np.array(
		[
			[cy, 0.0, sy],
			[0.0, 1.0, 0.0],
			[-sy, 0.0, cy],
		],
		dtype=np.float64,
	)
	rz_m = np.array(
		[
			[cz, -sz, 0.0],
			[sz, cz, 0.0],
			[0.0, 0.0, 1.0],
		],
		dtype=np.float64,
	)

	return rz_m @ ry_m @ rx_m


def _noise_settings_from_level(noise_level: float, noise_seed: Optional[int]) -> Optional[NoiseSettings]:
	level = float(noise_level)
	if level < 0.0:
		raise ValueError("noise_level must be >= 0.")
	if level == 0.0:
		return None

	# Keep read noise lower than shot noise so signal-dependent effects dominate.
	return NoiseSettings(
		shot_noise_strength=level,
		read_noise_std=0.25 * level,
		dark_bias=0.0,
		vignette_strength=min(0.08 + (0.4 * level), 0.35),
		seed=noise_seed,
	)


def _camera_axes_from_rotation(
	camera_rotation_matrix: np.ndarray,
	rotation_convention: Literal["world_to_camera", "camera_to_world"],
) -> tuple[np.ndarray, np.ndarray]:
	r = _validate_rotation_matrix(camera_rotation_matrix)

	if rotation_convention == "world_to_camera":
		r_cw = r
	elif rotation_convention == "camera_to_world":
		r_cw = r.T
	else:
		raise ValueError("rotation_convention must be 'world_to_camera' or 'camera_to_world'.")

	# Rows of R_cw are camera axes expressed in world frame.
	up_world = r_cw[1, :]
	forward_world = r_cw[2, :]
	return forward_world, up_world


def _normalize(vector: np.ndarray) -> np.ndarray:
	v = np.asarray(vector, dtype=np.float64)
	norm = float(np.linalg.norm(v))
	if norm <= 0.0:
		raise ValueError("Zero-length vector cannot be normalized.")
	return v / norm


def _nominal_center_pointing_rotation_cw(
	spacecraft_position_world: np.ndarray,
	body_center_world: np.ndarray,
	up_hint_world: np.ndarray,
) -> np.ndarray:
	to_center_world = _normalize(np.asarray(body_center_world, dtype=np.float64) - np.asarray(spacecraft_position_world, dtype=np.float64))
	up_hint = _normalize(up_hint_world)
	right_world = np.cross(to_center_world, up_hint)
	if np.linalg.norm(right_world) < 1e-9:
		fallback = np.array([1.0, 0.0, 0.0], dtype=np.float64)
		right_world = np.cross(to_center_world, fallback)
		if np.linalg.norm(right_world) < 1e-9:
			fallback = np.array([0.0, 0.0, 1.0], dtype=np.float64)
			right_world = np.cross(to_center_world, fallback)

	right_world = _normalize(right_world)
	up_world = _normalize(np.cross(right_world, to_center_world))
	forward_world = to_center_world

	# Rows are camera axes expressed in world frame: [x_cam; y_cam; z_cam].
	return np.stack((right_world, up_world, forward_world), axis=0)


def _resolve_spacecraft_position_world(
	body_center_world: np.ndarray,
	spacecraft_position_world: Optional[np.ndarray],
	spacecraft_position_relative_to_body_center: Optional[np.ndarray],
) -> np.ndarray:
	has_world = spacecraft_position_world is not None
	has_relative = spacecraft_position_relative_to_body_center is not None
	if has_world == has_relative:
		raise ValueError(
			"Provide exactly one of spacecraft_position_world or "
			"spacecraft_position_relative_to_body_center."
		)

	if has_world:
		return np.asarray(spacecraft_position_world, dtype=np.float64)

	return np.asarray(body_center_world, dtype=np.float64) + np.asarray(spacecraft_position_relative_to_body_center, dtype=np.float64)


def _resolve_camera_axes(
	spacecraft_position_world: np.ndarray,
	body_center_world: np.ndarray,
	camera_rotation_matrix: Optional[np.ndarray],
	rotation_convention: Literal["world_to_camera", "camera_to_world"],
	attitude_offset_from_center_pointing: Optional[np.ndarray],
	attitude_offset_convention: Literal["world_to_camera", "camera_to_world"],
	center_pointing_up_hint_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
	has_absolute_rotation = camera_rotation_matrix is not None
	has_relative_attitude = attitude_offset_from_center_pointing is not None

	if has_absolute_rotation and has_relative_attitude:
		raise ValueError(
			"Provide either camera_rotation_matrix or attitude_offset_from_center_pointing, not both."
		)

	if has_absolute_rotation:
		return _camera_axes_from_rotation(
			camera_rotation_matrix=np.asarray(camera_rotation_matrix, dtype=np.float64),
			rotation_convention=rotation_convention,
		)

	r_nominal_cw = _nominal_center_pointing_rotation_cw(
		spacecraft_position_world=spacecraft_position_world,
		body_center_world=body_center_world,
		up_hint_world=center_pointing_up_hint_world,
	)

	if not has_relative_attitude:
		r_cw = r_nominal_cw
	else:
		r_offset = _validate_rotation_matrix(np.asarray(attitude_offset_from_center_pointing, dtype=np.float64))
		if attitude_offset_convention == "world_to_camera":
			r_delta_cw = r_offset
		elif attitude_offset_convention == "camera_to_world":
			r_delta_cw = r_offset.T
		else:
			raise ValueError("attitude_offset_convention must be 'world_to_camera' or 'camera_to_world'.")

		# Relative attitude is applied on top of center-pointing orientation.
		r_cw = r_delta_cw @ r_nominal_cw

	up_world = r_cw[1, :]
	forward_world = r_cw[2, :]
	return forward_world, up_world


def state_to_edgepoints(
	spacecraft_position_world: Optional[np.ndarray] = None,
	camera_rotation_matrix: Optional[np.ndarray] = None,
	k_matrix: Optional[np.ndarray] = None,
	width: Optional[int] = None,
	height: Optional[int] = None,
	noise_level: Optional[float] = None,
	spacecraft_position_relative_to_body_center: Optional[np.ndarray] = None,
	attitude_offset_from_center_pointing: Optional[np.ndarray] = None,
	body_center_world: Optional[np.ndarray] = None,
	body_radius: float = 1_737.4,
	body_albedo_rgb: tuple[int, int, int] = (185, 185, 180),
	rotation_convention: Literal["world_to_camera", "camera_to_world"] = "world_to_camera",
	attitude_offset_convention: Literal["world_to_camera", "camera_to_world"] = "world_to_camera",
	center_pointing_up_hint_world: Optional[np.ndarray] = None,
	background_rgb: tuple[int, int, int] = (8, 10, 18),
	light_direction_world: Optional[np.ndarray] = None,
	ambient: float = 0.1,
	diffuse_strength: float = 0.9,
	limb_darkening_strength: float = 0.4,
	limb_darkening_power: float = 1.2,
	color_tolerance: float = 8.0,
	keep_largest_component_only: bool = True,
	fill_internal_holes: bool = False,
	remove_speckles: bool = False,
	min_speckle_neighbors: int = 3,
	speckle_iterations: int = 1,
	noise_seed: Optional[int] = None,
) -> StateToEdgePointsResult:
	"""Render a body from spacecraft state and return detected edge pixels.

	Inputs:
	- spacecraft_position_world or spacecraft_position_relative_to_body_center: choose one.
	- camera_rotation_matrix (absolute) or attitude_offset_from_center_pointing (relative): choose one.
	  If both are omitted, camera is center-pointing.
	  
	IMPORTANT: attitude_offset_from_center_pointing conventions:
	  - The "center-pointing" frame has Z-axis pointing from spacecraft to planet center (spacecraft body frame).
	  - attitude_offset_from_center_pointing is a rotation matrix representing the offset from this frame.
	  - With attitude_offset_convention="world_to_camera" (default):
	    * The matrix represents rotation from spacecraft body frame to camera frame.
	    * Euler angles [x, y, z] describe camera boresight relative to spacecraft Z-axis (pointing at planet).
	  - With attitude_offset_convention="camera_to_world":
	    * The matrix represents rotation from camera frame to spacecraft body frame (transpose of above).
	    
	- k_matrix: intrinsic camera matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]].
	- width, height: image dimensions.
	- noise_level: scalar noise strength (0 disables noise).
	- All geometric quantities are expected to be in kilometers (km).
	"""
	if k_matrix is None:
		raise ValueError("k_matrix is required.")
	if width is None or height is None:
		raise ValueError("width and height are required.")
	if noise_level is None:
		raise ValueError("noise_level is required.")
	if int(width) <= 0 or int(height) <= 0:
		raise ValueError("width and height must be > 0.")
	if body_center_world is None:
		body_center_world = np.array([0.0, 0.0, 0.0], dtype=np.float64)
	if center_pointing_up_hint_world is None:
		center_pointing_up_hint_world = np.array([0.0, 1.0, 0.0], dtype=np.float64)

	spacecraft_position_resolved = _resolve_spacecraft_position_world(
		body_center_world=np.asarray(body_center_world, dtype=np.float64),
		spacecraft_position_world=spacecraft_position_world,
		spacecraft_position_relative_to_body_center=spacecraft_position_relative_to_body_center,
	)

	forward_world, up_world = _resolve_camera_axes(
		spacecraft_position_world=spacecraft_position_resolved,
		body_center_world=np.asarray(body_center_world, dtype=np.float64),
		camera_rotation_matrix=None if camera_rotation_matrix is None else np.asarray(camera_rotation_matrix, dtype=np.float64),
		rotation_convention=rotation_convention,
		attitude_offset_from_center_pointing=None
		if attitude_offset_from_center_pointing is None
		else np.asarray(attitude_offset_from_center_pointing, dtype=np.float64),
		attitude_offset_convention=attitude_offset_convention,
		center_pointing_up_hint_world=np.asarray(center_pointing_up_hint_world, dtype=np.float64),
	)

	camera = CameraSettings(
		width=int(width),
		height=int(height),
		horizontal_fov_deg=None,
		position_world=spacecraft_position_resolved,
		forward_world=np.asarray(forward_world, dtype=np.float64),
		up_world=np.asarray(up_world, dtype=np.float64),
		k_matrix=np.asarray(k_matrix, dtype=np.float64),
	)

	body = CelestialBody(
		center_world=np.asarray(body_center_world, dtype=np.float64),
		radius=float(body_radius),
		albedo_rgb=body_albedo_rgb,
	)

	noise_settings = _noise_settings_from_level(noise_level=noise_level, noise_seed=noise_seed)

	render = render_celestial_body(
		camera=camera,
		body=body,
		background_rgb=background_rgb,
		light_direction_world=None if light_direction_world is None else np.asarray(light_direction_world, dtype=np.float64),
		ambient=float(ambient),
		diffuse_strength=float(diffuse_strength),
		limb_darkening_strength=float(limb_darkening_strength),
		limb_darkening_power=float(limb_darkening_power),
		noise_settings=noise_settings,
	)

	edges = find_body_edge_pixels(
		image_rgb=render.image,
		background_rgb=background_rgb,
		color_tolerance=float(color_tolerance),
		keep_largest_component_only=bool(keep_largest_component_only),
		fill_internal_holes=bool(fill_internal_holes),
		remove_speckles=bool(remove_speckles),
		min_speckle_neighbors=int(min_speckle_neighbors),
		speckle_iterations=int(speckle_iterations),
	)

	return StateToEdgePointsResult(render=render, edges=edges, camera=camera)


if __name__ == "__main__":
	w = 1280
	h = 720
	K = np.array([[1120.0, 0.0, w / 2.0], [0.0, 1120.0, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)

	# Identity means camera axes align with world axes in this demo.
	R_wc = np.eye(3, dtype=np.float64)

	out = state_to_edgepoints(
		spacecraft_position_relative_to_body_center=np.array([0.0, 0.0, 4_000.0], dtype=np.float64),
		attitude_offset_from_center_pointing=np.eye(3, dtype=np.float64),
		k_matrix=K,
		width=w,
		height=h,
		noise_level=0.04,
	)

	print(f"Render center: ({out.render.center_px[0]:.2f}, {out.render.center_px[1]:.2f})")
	print(f"Detected edge pixels: {len(out.edges.edge_coordinates_xy)}")
