from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Sequence
import json

import numpy as np


@dataclass(frozen=True)
class CameraSettings:
	width: int
	height: int
	horizontal_fov_deg: Optional[float]
	position_world: np.ndarray
	forward_world: np.ndarray
	up_world: np.ndarray
	k_matrix: Optional[np.ndarray] = None


@dataclass(frozen=True)
class CelestialBody:
	center_world: np.ndarray
	radius: float
	albedo_rgb: tuple[int, int, int] = (170, 170, 170)


@dataclass(frozen=True)
class RenderResult:
	image: np.ndarray
	center_px: tuple[float, float]
	radius_px_x: float
	radius_px_y: float
	angular_diameter_deg: float


@dataclass(frozen=True)
class NoiseSettings:
	shot_noise_strength: float = 0.04
	read_noise_std: float = 0.01
	dark_bias: float = 0.0
	vignette_strength: float = 0.08
	seed: Optional[int] = None


def _normalize(vector: np.ndarray) -> np.ndarray:
	v = np.asarray(vector, dtype=np.float64)
	norm = np.linalg.norm(v)
	if norm == 0:
		raise ValueError("Zero-length vector cannot be normalized.")
	return v / norm


def _camera_basis(camera: CameraSettings) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	forward = _normalize(camera.forward_world)
	up_hint = _normalize(camera.up_world)
	right = _normalize(np.cross(forward, up_hint))
	up = _normalize(np.cross(right, forward))
	return right, up, forward


def _focal_lengths(width: int, height: int, horizontal_fov_deg: float) -> tuple[float, float]:
	hfov_rad = np.deg2rad(horizontal_fov_deg)
	if hfov_rad <= 0 or hfov_rad >= np.pi:
		raise ValueError("horizontal_fov_deg must be in the range (0, 180).")

	fx = width / (2.0 * np.tan(hfov_rad / 2.0))
	vfov_rad = 2.0 * np.arctan((height / width) * np.tan(hfov_rad / 2.0))
	fy = height / (2.0 * np.tan(vfov_rad / 2.0))
	return fx, fy


def _camera_intrinsics(camera: CameraSettings) -> tuple[float, float, float, float]:
	if camera.k_matrix is not None:
		k = np.asarray(camera.k_matrix, dtype=np.float64)
		if k.shape != (3, 3):
			raise ValueError("k_matrix must have shape (3, 3).")
		fx = float(k[0, 0])
		fy = float(k[1, 1])
		cx = float(k[0, 2])
		cy = float(k[1, 2])
		if fx <= 0.0 or fy <= 0.0:
			raise ValueError("k_matrix focal lengths K[0,0] and K[1,1] must be > 0.")
		return fx, fy, cx, cy

	if camera.horizontal_fov_deg is None:
		raise ValueError("Either k_matrix or horizontal_fov_deg must be provided.")

	fx, fy = _focal_lengths(camera.width, camera.height, float(camera.horizontal_fov_deg))
	cx = camera.width / 2.0
	cy = camera.height / 2.0
	return fx, fy, cx, cy


def _camera_fovs_deg(camera: CameraSettings) -> tuple[float, float]:
	if camera.k_matrix is not None:
		k = np.asarray(camera.k_matrix, dtype=np.float64)
		if k.shape != (3, 3):
			raise ValueError("k_matrix must have shape (3, 3).")
		fx = float(k[0, 0])
		fy = float(k[1, 1])
		if fx <= 0.0 or fy <= 0.0:
			raise ValueError("k_matrix focal lengths K[0,0] and K[1,1] must be > 0.")
		hfov_rad = 2.0 * np.arctan(camera.width / (2.0 * fx))
		vfov_rad = 2.0 * np.arctan(camera.height / (2.0 * fy))
		return float(np.rad2deg(hfov_rad)), float(np.rad2deg(vfov_rad))

	if camera.horizontal_fov_deg is None:
		raise ValueError("Either k_matrix or horizontal_fov_deg must be provided.")

	hfov_deg = float(camera.horizontal_fov_deg)
	vfov_deg = _vertical_fov_deg(camera.width, camera.height, hfov_deg)
	return hfov_deg, vfov_deg


def _apply_realistic_noise(image_uint8: np.ndarray, noise: NoiseSettings) -> np.ndarray:
	img = np.asarray(image_uint8, dtype=np.float64) / 255.0
	rng = np.random.default_rng(noise.seed)

	if noise.shot_noise_strength > 0.0:
		shot_sigma = noise.shot_noise_strength * np.sqrt(np.clip(img, 0.0, 1.0) + 1e-8)
		img = img + rng.normal(loc=0.0, scale=shot_sigma, size=img.shape)

	if noise.read_noise_std > 0.0:
		img = img + rng.normal(loc=0.0, scale=noise.read_noise_std, size=img.shape)

	if noise.vignette_strength > 0.0:
		h, w = img.shape[:2]
		yy, xx = np.indices((h, w), dtype=np.float64)
		cx = (w - 1) / 2.0
		cy = (h - 1) / 2.0
		nx = (xx - cx) / max(cx, 1.0)
		ny = (yy - cy) / max(cy, 1.0)
		r2 = np.clip(nx * nx + ny * ny, 0.0, 1.0)
		vignette = 1.0 - noise.vignette_strength * r2
		img = img * vignette[..., None]

	img = img + float(noise.dark_bias)
	img = np.clip(img, 0.0, 1.0)
	return (img * 255.0).astype(np.uint8)


def _vertical_fov_deg(width: int, height: int, horizontal_fov_deg: float) -> float:
	hfov_rad = np.deg2rad(horizontal_fov_deg)
	vfov_rad = 2.0 * np.arctan((height / width) * np.tan(hfov_rad / 2.0))
	return float(np.rad2deg(vfov_rad))


def _camera_up_from_forward(forward_world: np.ndarray, up_hint_world: np.ndarray) -> np.ndarray:
	forward = _normalize(forward_world)
	up_hint = _normalize(up_hint_world)
	right = np.cross(forward, up_hint)
	if np.linalg.norm(right) < 1e-9:
		fallback = np.array([1.0, 0.0, 0.0], dtype=np.float64)
		right = np.cross(forward, fallback)
		if np.linalg.norm(right) < 1e-9:
			fallback = np.array([0.0, 0.0, 1.0], dtype=np.float64)
			right = np.cross(forward, fallback)
	right = _normalize(right)
	up = _normalize(np.cross(right, forward))
	return up


def _camera_right_up_from_forward(forward_world: np.ndarray, up_hint_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	forward = _normalize(forward_world)
	up = _camera_up_from_forward(forward, up_hint_world)
	right = _normalize(np.cross(forward, up))
	return right, up


def _build_look_at_camera(
	template_camera: CameraSettings,
	position_world: np.ndarray,
	look_target_world: np.ndarray,
	forward_offset_world: Optional[np.ndarray] = None,
) -> CameraSettings:
	position = np.asarray(position_world, dtype=np.float64)
	target = np.asarray(look_target_world, dtype=np.float64)
	nominal_forward = _normalize(target - position)

	if forward_offset_world is None:
		forward = nominal_forward
	else:
		offset = np.asarray(forward_offset_world, dtype=np.float64)
		forward = _normalize(nominal_forward + offset)

	up = _camera_up_from_forward(forward, np.asarray(template_camera.up_world, dtype=np.float64))
	return CameraSettings(
		width=template_camera.width,
		height=template_camera.height,
		horizontal_fov_deg=template_camera.horizontal_fov_deg,
		k_matrix=None if template_camera.k_matrix is None else np.asarray(template_camera.k_matrix, dtype=np.float64).copy(),
		position_world=position,
		forward_world=forward,
		up_world=up,
	)


def render_celestial_body(
	camera: CameraSettings,
	body: CelestialBody,
	background_rgb: tuple[int, int, int] = (0, 0, 0),
	light_direction_world: Optional[np.ndarray] = None,
	ambient: float = 0.12,
	diffuse_strength: float = 0.88,
	limb_darkening_strength: float = 0.35,
	limb_darkening_power: float = 1.2,
	noise_settings: Optional[NoiseSettings] = None,
) -> RenderResult:
	width = int(camera.width)
	height = int(camera.height)
	if width <= 0 or height <= 0:
		raise ValueError("Image width and height must both be > 0.")

	cam_pos = np.asarray(camera.position_world, dtype=np.float64)
	body_center = np.asarray(body.center_world, dtype=np.float64)
	radius = float(body.radius)
	if radius <= 0:
		raise ValueError("Body radius must be > 0.")

	right, up, forward = _camera_basis(camera)
	fx, fy, cx, cy = _camera_intrinsics(camera)

	to_center_world = body_center - cam_pos
	distance = float(np.linalg.norm(to_center_world))
	if distance <= radius:
		raise ValueError("Camera must be outside the celestial body (distance > radius).")

	center_cam_x = float(np.dot(to_center_world, right))
	center_cam_y = float(np.dot(to_center_world, up))
	center_cam_z = float(np.dot(to_center_world, forward))

	if center_cam_z <= 0:
		raise ValueError("Celestial body center is behind the camera.")

	center_px_x = cx + (fx * center_cam_x / center_cam_z)
	center_px_y = cy - (fy * center_cam_y / center_cam_z)

	angular_radius_rad = float(np.arcsin(radius / distance))
	angular_diameter_deg = float(np.rad2deg(2.0 * angular_radius_rad))

	radius_px_x = float(fx * np.tan(angular_radius_rad))
	radius_px_y = float(fy * np.tan(angular_radius_rad))

	image = np.zeros((height, width, 3), dtype=np.float64)
	image[:, :] = np.asarray(background_rgb, dtype=np.float64)

	yy, xx = np.indices((height, width), dtype=np.float64)
	nx = (xx - center_px_x) / max(radius_px_x, 1e-12)
	ny = (yy - center_px_y) / max(radius_px_y, 1e-12)
	r2 = nx * nx + ny * ny
	inside = r2 <= 1.0

	if np.any(inside):
		nz = np.zeros_like(r2)
		nz[inside] = np.sqrt(1.0 - r2[inside])

		normals_cam = np.stack((nx, -ny, nz), axis=-1)

		if light_direction_world is None:
			to_camera_world = cam_pos - body_center
			light_world = _normalize(to_camera_world)
		else:
			light_world = _normalize(np.asarray(light_direction_world, dtype=np.float64))

		light_cam = np.array(
			[
				float(np.dot(light_world, right)),
				float(np.dot(light_world, up)),
				float(np.dot(light_world, forward)),
			],
			dtype=np.float64,
		)
		light_cam = _normalize(light_cam)

		lambert = np.clip(
			normals_cam[..., 0] * light_cam[0]
			+ normals_cam[..., 1] * light_cam[1]
			+ normals_cam[..., 2] * light_cam[2],
			0.0,
			1.0,
		)

		limb = np.power(np.clip(nz, 0.0, 1.0), limb_darkening_power)
		limb_term = (1.0 - limb_darkening_strength) + (limb_darkening_strength * limb)

		intensity = (ambient + diffuse_strength * lambert) * limb_term
		intensity = np.clip(intensity, 0.0, 1.0)

		base_color = np.asarray(body.albedo_rgb, dtype=np.float64)
		shaded = intensity[..., None] * base_color[None, None, :]
		image[inside] = shaded[inside]

	image_uint8 = np.clip(image, 0.0, 255.0).astype(np.uint8)
	if noise_settings is not None:
		image_uint8 = _apply_realistic_noise(image_uint8, noise_settings)

	return RenderResult(
		image=image_uint8,
		center_px=(center_px_x, center_px_y),
		radius_px_x=radius_px_x,
		radius_px_y=radius_px_y,
		angular_diameter_deg=angular_diameter_deg,
	)


def estimate_body_pixel_diameter(
	camera: CameraSettings,
	body_center_world: np.ndarray,
	body_radius: float,
) -> tuple[float, float, float]:
	cam_pos = np.asarray(camera.position_world, dtype=np.float64)
	distance = float(np.linalg.norm(np.asarray(body_center_world, dtype=np.float64) - cam_pos))
	if distance <= body_radius:
		raise ValueError("Camera must be outside the celestial body (distance > radius).")

	fx, fy, _, _ = _camera_intrinsics(camera)
	angular_radius_rad = float(np.arcsin(body_radius / distance))

	diameter_x = 2.0 * fx * np.tan(angular_radius_rad)
	diameter_y = 2.0 * fy * np.tan(angular_radius_rad)
	diameter_deg = float(np.rad2deg(2.0 * angular_radius_rad))
	return float(diameter_x), float(diameter_y), diameter_deg


def minimum_distance_for_full_visibility(
	body_radius: float,
	camera: CameraSettings,
	safety_margin_deg: float = 0.25,
) -> float:
	if body_radius <= 0:
		raise ValueError("body_radius must be > 0.")

	hfov_deg, vfov_deg = _camera_fovs_deg(camera)
	half_min_fov_deg = 0.5 * min(hfov_deg, vfov_deg)
	allowed_angular_radius_deg = half_min_fov_deg - safety_margin_deg
	if allowed_angular_radius_deg <= 0.0:
		raise ValueError("Camera FOV is too narrow for the requested safety margin.")

	allowed_angular_radius_rad = np.deg2rad(allowed_angular_radius_deg)
	return float(body_radius / np.sin(allowed_angular_radius_rad))


def generate_images_from_positions_and_offsets(
	template_camera: CameraSettings,
	body: CelestialBody,
	position_vectors: Sequence[np.ndarray],
	camera_offset_vectors: Sequence[np.ndarray],
	background_rgb: tuple[int, int, int] = (0, 0, 0),
	light_direction_world: Optional[np.ndarray] = None,
	ambient: float = 0.12,
	diffuse_strength: float = 0.88,
	limb_darkening_strength: float = 0.35,
	limb_darkening_power: float = 1.2,
	noise_settings: Optional[NoiseSettings] = None,
) -> list[RenderResult]:
	if len(position_vectors) != len(camera_offset_vectors):
		raise ValueError("position_vectors and camera_offset_vectors must be the same length.")

	results: list[RenderResult] = []
	for position_world, forward_offset in zip(position_vectors, camera_offset_vectors):
		camera_i = _build_look_at_camera(
			template_camera=template_camera,
			position_world=np.asarray(position_world, dtype=np.float64),
			look_target_world=np.asarray(body.center_world, dtype=np.float64),
			forward_offset_world=np.asarray(forward_offset, dtype=np.float64),
		)

		result_i = render_celestial_body(
			camera=camera_i,
			body=body,
			background_rgb=background_rgb,
			light_direction_world=light_direction_world,
			ambient=ambient,
			diffuse_strength=diffuse_strength,
			limb_darkening_strength=limb_darkening_strength,
			limb_darkening_power=limb_darkening_power,
			noise_settings=noise_settings,
		)
		results.append(result_i)

	return results


def generate_visible_position_and_offset_vectors(
	body: CelestialBody,
	template_camera: CameraSettings,
	sample_count: int,
	min_distance: float,
	max_distance: float,
	max_pointing_offset_fraction: float = 0.8,
	safety_margin_deg: float = 0.25,
	visibility_mode: str = "full",
	seed: Optional[int] = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[tuple[float, float]]]:
	if sample_count <= 0:
		raise ValueError("sample_count must be > 0.")
	if min_distance <= body.radius:
		raise ValueError("min_distance must be greater than body.radius.")
	if max_distance < min_distance:
		raise ValueError("max_distance must be >= min_distance.")
	if not (0.0 <= max_pointing_offset_fraction <= 1.0):
		raise ValueError("max_pointing_offset_fraction must be in [0, 1].")
	if visibility_mode not in ("full", "limb"):
		raise ValueError("visibility_mode must be either 'full' or 'limb'.")

	rng = np.random.default_rng(seed)
	body_center = np.asarray(body.center_world, dtype=np.float64)
	up_hint = np.asarray(template_camera.up_world, dtype=np.float64)

	if visibility_mode == "full":
		min_visible_distance = minimum_distance_for_full_visibility(
			body_radius=body.radius,
			camera=template_camera,
			safety_margin_deg=safety_margin_deg,
		)
		effective_min_distance = max(min_distance, min_visible_distance)
		if max_distance < effective_min_distance:
			raise ValueError(
				"Provided distance range cannot keep the full body in frame. "
				f"Need max_distance >= {effective_min_distance:.3f}"
			)
	else:
		effective_min_distance = min_distance

	positions: list[np.ndarray] = []
	offsets: list[np.ndarray] = []
	angles_deg: list[tuple[float, float]] = []

	for _ in range(sample_count):
		distance = float(rng.uniform(effective_min_distance, max_distance))
		angular_radius_deg = float(np.rad2deg(np.arcsin(body.radius / distance)))

		hfov_deg, vfov_deg = _camera_fovs_deg(template_camera)
		half_min_fov_deg = 0.5 * min(hfov_deg, vfov_deg)
		if visibility_mode == "full":
			min_center_offset_deg = 0.0
			max_center_offset_deg = max(0.0, half_min_fov_deg - angular_radius_deg - safety_margin_deg)
		else:
			min_center_offset_deg = max(0.0, angular_radius_deg - half_min_fov_deg + safety_margin_deg)
			max_center_offset_deg = max(0.0, half_min_fov_deg + angular_radius_deg - safety_margin_deg)

		if max_center_offset_deg <= 0.0:
			raise ValueError("No valid pointing offsets found for the provided visibility constraints.")

		position_dir = _normalize(rng.normal(size=3))
		position_world = body_center + position_dir * distance

		nominal_forward = _normalize(body_center - position_world)
		right, up = _camera_right_up_from_forward(nominal_forward, up_hint)

		max_offset_deg = max_center_offset_deg * max_pointing_offset_fraction
		max_offset_deg = max(max_offset_deg, min_center_offset_deg)
		if max_offset_deg <= 0.0:
			yaw_deg = 0.0
			pitch_deg = 0.0
		else:
			if max_offset_deg > min_center_offset_deg:
				u = rng.random()
				r = np.sqrt((u * (max_offset_deg**2 - min_center_offset_deg**2)) + min_center_offset_deg**2)
			else:
				r = min_center_offset_deg
			phi = 2.0 * np.pi * rng.random()
			yaw_deg = float(r * np.cos(phi))
			pitch_deg = float(r * np.sin(phi))

		yaw_rad = np.deg2rad(yaw_deg)
		pitch_rad = np.deg2rad(pitch_deg)
		tilt = (np.tan(yaw_rad) * right) + (np.tan(pitch_rad) * up)
		pointed_forward = _normalize(nominal_forward + tilt)
		offset_world = pointed_forward - nominal_forward

		positions.append(position_world)
		offsets.append(offset_world)
		angles_deg.append((yaw_deg, pitch_deg))

	return positions, offsets, angles_deg


def generate_images_with_randomized_views(
	body: CelestialBody,
	template_camera: CameraSettings,
	sample_count: int,
	min_distance: float,
	max_distance: float,
	max_pointing_offset_fraction: float = 0.8,
	safety_margin_deg: float = 0.25,
	visibility_mode: str = "full",
	seed: Optional[int] = None,
	background_rgb: tuple[int, int, int] = (0, 0, 0),
	light_direction_world: Optional[np.ndarray] = None,
	ambient: float = 0.12,
	diffuse_strength: float = 0.88,
	limb_darkening_strength: float = 0.35,
	limb_darkening_power: float = 1.2,
	noise_settings: Optional[NoiseSettings] = None,
) -> tuple[list[RenderResult], list[np.ndarray], list[np.ndarray], list[tuple[float, float]]]:
	positions, offsets, angles_deg = generate_visible_position_and_offset_vectors(
		body=body,
		template_camera=template_camera,
		sample_count=sample_count,
		min_distance=min_distance,
		max_distance=max_distance,
		max_pointing_offset_fraction=max_pointing_offset_fraction,
		safety_margin_deg=safety_margin_deg,
		visibility_mode=visibility_mode,
		seed=seed,
	)

	results = generate_images_from_positions_and_offsets(
		template_camera=template_camera,
		body=body,
		position_vectors=positions,
		camera_offset_vectors=offsets,
		background_rgb=background_rgb,
		light_direction_world=light_direction_world,
		ambient=ambient,
		diffuse_strength=diffuse_strength,
		limb_darkening_strength=limb_darkening_strength,
		limb_darkening_power=limb_darkening_power,
		noise_settings=noise_settings,
	)

	return results, positions, offsets, angles_deg


def save_render_results_with_metadata(
	results: Sequence[RenderResult],
	output_dir: str | Path = "Body_Images",
	file_prefix: str = "body",
	camera: Optional[CameraSettings] = None,
	body: Optional[CelestialBody] = None,
	noise_settings: Optional[NoiseSettings] = None,
	positions: Optional[Sequence[np.ndarray]] = None,
	offsets: Optional[Sequence[np.ndarray]] = None,
	angles_deg: Optional[Sequence[tuple[float, float]]] = None,
	append_manifest: bool = True,
) -> tuple[list[Path], Path]:
	if len(results) == 0:
		raise ValueError("results cannot be empty.")

	output_path = Path(output_dir)
	output_path.mkdir(parents=True, exist_ok=True)
	manifest_path = output_path / "manifest.jsonl"

	if positions is not None and len(positions) != len(results):
		raise ValueError("positions length must match results length.")
	if offsets is not None and len(offsets) != len(results):
		raise ValueError("offsets length must match results length.")
	if angles_deg is not None and len(angles_deg) != len(results):
		raise ValueError("angles_deg length must match results length.")

	try:
		from PIL import Image
	except Exception as exc:
		raise ImportError("Pillow is required to save PNG images. Install with `pip install pillow`.") from exc

	open_mode = "a" if append_manifest else "w"
	saved_paths: list[Path] = []

	with manifest_path.open(open_mode, encoding="utf-8") as manifest:
		for index, result in enumerate(results):
			image_name = f"{file_prefix}_{index:05d}.png"
			image_path = output_path / image_name
			Image.fromarray(result.image).save(image_path)
			saved_paths.append(image_path)

			record: dict[str, object] = {
				"image_file": image_name,
				"center_px": [float(result.center_px[0]), float(result.center_px[1])],
				"radius_px": [float(result.radius_px_x), float(result.radius_px_y)],
				"angular_diameter_deg": float(result.angular_diameter_deg),
				"image_shape": [int(result.image.shape[0]), int(result.image.shape[1]), int(result.image.shape[2])],
			}

			if camera is not None:
				record["camera"] = {
					"width": int(camera.width),
					"height": int(camera.height),
					"horizontal_fov_deg": None if camera.horizontal_fov_deg is None else float(camera.horizontal_fov_deg),
					"k_matrix": None if camera.k_matrix is None else np.asarray(camera.k_matrix, dtype=np.float64).tolist(),
					"position_world": np.asarray(camera.position_world, dtype=np.float64).tolist(),
					"forward_world": np.asarray(camera.forward_world, dtype=np.float64).tolist(),
					"up_world": np.asarray(camera.up_world, dtype=np.float64).tolist(),
				}

			if body is not None:
				record["body"] = {
					"center_world": np.asarray(body.center_world, dtype=np.float64).tolist(),
					"radius": float(body.radius),
					"albedo_rgb": [int(body.albedo_rgb[0]), int(body.albedo_rgb[1]), int(body.albedo_rgb[2])],
				}

			if noise_settings is not None:
				record["noise_settings"] = asdict(noise_settings)

			if positions is not None:
				record["sample_position_world"] = np.asarray(positions[index], dtype=np.float64).tolist()
			if offsets is not None:
				record["sample_forward_offset_world"] = np.asarray(offsets[index], dtype=np.float64).tolist()
			if angles_deg is not None:
				record["sample_angles_deg"] = [float(angles_deg[index][0]), float(angles_deg[index][1])]

			manifest.write(json.dumps(record) + "\n")

	return saved_paths, manifest_path


def generate_and_save_randomized_dataset(
	body: CelestialBody,
	template_camera: CameraSettings,
	sample_count: int,
	min_distance: float,
	max_distance: float,
	output_dir: str | Path = "Body_Images",
	file_prefix: str = "body",
	max_pointing_offset_fraction: float = 0.8,
	safety_margin_deg: float = 0.25,
	visibility_mode: str = "full",
	seed: Optional[int] = None,
	background_rgb: tuple[int, int, int] = (0, 0, 0),
	light_direction_world: Optional[np.ndarray] = None,
	ambient: float = 0.12,
	diffuse_strength: float = 0.88,
	limb_darkening_strength: float = 0.35,
	limb_darkening_power: float = 1.2,
	noise_settings: Optional[NoiseSettings] = None,
	append_manifest: bool = True,
) -> tuple[list[RenderResult], list[Path], Path]:
	results, positions, offsets, angles_deg = generate_images_with_randomized_views(
		body=body,
		template_camera=template_camera,
		sample_count=sample_count,
		min_distance=min_distance,
		max_distance=max_distance,
		max_pointing_offset_fraction=max_pointing_offset_fraction,
		safety_margin_deg=safety_margin_deg,
		visibility_mode=visibility_mode,
		seed=seed,
		background_rgb=background_rgb,
		light_direction_world=light_direction_world,
		ambient=ambient,
		diffuse_strength=diffuse_strength,
		limb_darkening_strength=limb_darkening_strength,
		limb_darkening_power=limb_darkening_power,
		noise_settings=noise_settings,
	)

	saved_paths, manifest_path = save_render_results_with_metadata(
		results=results,
		output_dir=output_dir,
		file_prefix=file_prefix,
		camera=template_camera,
		body=body,
		noise_settings=noise_settings,
		positions=positions,
		offsets=offsets,
		angles_deg=angles_deg,
		append_manifest=append_manifest,
	)

	return results, saved_paths, manifest_path


if __name__ == "__main__":
	camera = CameraSettings(
		width=1280,
		height=720,
		horizontal_fov_deg=60.0,  # Or provide k_matrix=np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).
		position_world=np.array([0.0, 0.0, 4_000.0]),
		forward_world=np.array([0.0, 0.0, -1.0]),
		up_world=np.array([0.0, 1.0, 0.0]),
	)

	moon_like = CelestialBody(
		center_world=np.array([0.0, 0.0, 0.0]),
		radius=1_737.4,
		albedo_rgb=(185, 185, 180),
	)

	result = render_celestial_body(
		camera=camera,
		body=moon_like,
		background_rgb=(8, 10, 18),
		light_direction_world=np.array([1.0, 0.2, 0.6]),
		ambient=0.1,
		diffuse_strength=0.9,
		limb_darkening_strength=0.4,
	)

	print(f"Angular diameter: {result.angular_diameter_deg:.3f} deg")
	print(f"Projected radii: x={result.radius_px_x:.2f}px, y={result.radius_px_y:.2f}px")

	try:
		from PIL import Image

		Image.fromarray(result.image).save("celestial_render.png")
		print("Saved: celestial_render.png")
	except Exception:
		print("Install Pillow to save output as PNG, or display result.image in your notebook.")
