from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class EdgeDetectionResult:
	edge_coordinates_xy: np.ndarray
	edge_mask: np.ndarray
	foreground_mask: np.ndarray


def load_image_rgb(image_path: str | Path) -> np.ndarray:
	path = Path(image_path)
	if not path.exists():
		raise FileNotFoundError(f"Image not found: {path}")

	try:
		from PIL import Image
	except Exception as exc:
		raise ImportError("Pillow is required to load image paths. Install with `pip install pillow`.") from exc

	image = Image.open(path).convert("RGB")
	return np.asarray(image, dtype=np.uint8)


def infer_background_color_from_border(image_rgb: np.ndarray) -> np.ndarray:
	img = np.asarray(image_rgb, dtype=np.uint8)
	if img.ndim != 3 or img.shape[2] != 3:
		raise ValueError("image_rgb must have shape (H, W, 3).")

	top = img[0, :, :]
	bottom = img[-1, :, :]
	left = img[:, 0, :]
	right = img[:, -1, :]
	border = np.concatenate([top, bottom, left, right], axis=0).astype(np.float64)
	return np.median(border, axis=0)


def build_foreground_mask(
	image_rgb: np.ndarray,
	background_rgb: Optional[np.ndarray | tuple[int, int, int]] = None,
	color_tolerance: float = 8.0,
) -> np.ndarray:
	img = np.asarray(image_rgb, dtype=np.float64)
	if img.ndim != 3 or img.shape[2] != 3:
		raise ValueError("image_rgb must have shape (H, W, 3).")

	if background_rgb is None:
		bg = infer_background_color_from_border(img)
	else:
		bg = np.asarray(background_rgb, dtype=np.float64)
		if bg.shape != (3,):
			raise ValueError("background_rgb must be length-3 RGB.")

	color_distance = np.linalg.norm(img - bg[None, None, :], axis=2)
	foreground_mask = color_distance > float(color_tolerance)
	return foreground_mask


def extract_edge_mask(foreground_mask: np.ndarray) -> np.ndarray:
	mask = np.asarray(foreground_mask, dtype=bool)
	if mask.ndim != 2:
		raise ValueError("foreground_mask must be a 2D boolean array.")

	edge = np.zeros_like(mask, dtype=bool)

	# Interior boundary pixels (4-neighborhood)
	interior = mask[1:-1, 1:-1]
	up = mask[:-2, 1:-1]
	down = mask[2:, 1:-1]
	left = mask[1:-1, :-2]
	right = mask[1:-1, 2:]
	edge[1:-1, 1:-1] = interior & (~up | ~down | ~left | ~right)

	# Outer image border handling
	edge[0, :] |= mask[0, :]
	edge[-1, :] |= mask[-1, :]
	edge[:, 0] |= mask[:, 0]
	edge[:, -1] |= mask[:, -1]

	return edge


def keep_largest_connected_component(mask: np.ndarray) -> np.ndarray:
	mask_bool = np.asarray(mask, dtype=bool)
	if mask_bool.ndim != 2:
		raise ValueError("mask must be a 2D boolean array.")

	height, width = mask_bool.shape
	visited = np.zeros_like(mask_bool, dtype=bool)
	best_component: list[tuple[int, int]] = []

	for y in range(height):
		for x in range(width):
			if not mask_bool[y, x] or visited[y, x]:
				continue

			stack = [(y, x)]
			visited[y, x] = True
			component: list[tuple[int, int]] = []

			while stack:
				cy, cx = stack.pop()
				component.append((cy, cx))

				for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
					if 0 <= ny < height and 0 <= nx < width:
						if mask_bool[ny, nx] and not visited[ny, nx]:
							visited[ny, nx] = True
							stack.append((ny, nx))

			if len(component) > len(best_component):
				best_component = component

	filtered = np.zeros_like(mask_bool, dtype=bool)
	for y, x in best_component:
		filtered[y, x] = True
	return filtered


def _fill_internal_holes(mask: np.ndarray) -> np.ndarray:
	mask_bool = np.asarray(mask, dtype=bool)
	height, width = mask_bool.shape

	background = ~mask_bool
	visited = np.zeros_like(background, dtype=bool)
	stack: list[tuple[int, int]] = []

	for x in range(width):
		if background[0, x] and not visited[0, x]:
			visited[0, x] = True
			stack.append((0, x))
		if background[height - 1, x] and not visited[height - 1, x]:
			visited[height - 1, x] = True
			stack.append((height - 1, x))

	for y in range(height):
		if background[y, 0] and not visited[y, 0]:
			visited[y, 0] = True
			stack.append((y, 0))
		if background[y, width - 1] and not visited[y, width - 1]:
			visited[y, width - 1] = True
			stack.append((y, width - 1))

	while stack:
		cy, cx = stack.pop()
		for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
			if 0 <= ny < height and 0 <= nx < width:
				if background[ny, nx] and not visited[ny, nx]:
					visited[ny, nx] = True
					stack.append((ny, nx))

	holes = background & (~visited)
	filled = mask_bool | holes
	return filled


def _remove_speckles(mask: np.ndarray, min_neighbors: int = 3, iterations: int = 1) -> np.ndarray:
	if min_neighbors < 0 or min_neighbors > 8:
		raise ValueError("min_neighbors must be in [0, 8].")
	if iterations < 0:
		raise ValueError("iterations must be >= 0.")

	cleaned = np.asarray(mask, dtype=bool).copy()
	for _ in range(iterations):
		neighbor_count = np.zeros_like(cleaned, dtype=np.int32)
		for dy in (-1, 0, 1):
			for dx in (-1, 0, 1):
				if dy == 0 and dx == 0:
					continue
				src_y0 = max(0, -dy)
				src_y1 = cleaned.shape[0] - max(0, dy)
				src_x0 = max(0, -dx)
				src_x1 = cleaned.shape[1] - max(0, dx)

				dst_y0 = max(0, dy)
				dst_y1 = cleaned.shape[0] - max(0, -dy)
				dst_x0 = max(0, dx)
				dst_x1 = cleaned.shape[1] - max(0, -dx)

				neighbor_count[dst_y0:dst_y1, dst_x0:dst_x1] += cleaned[src_y0:src_y1, src_x0:src_x1]

		cleaned = cleaned & (neighbor_count >= min_neighbors)
	return cleaned


def edge_coordinates_from_mask(edge_mask: np.ndarray) -> np.ndarray:
	edge = np.asarray(edge_mask, dtype=bool)
	y_idx, x_idx = np.where(edge)
	return np.column_stack((x_idx, y_idx)).astype(np.int32)


def find_body_edge_pixels(
	image_rgb: np.ndarray,
	background_rgb: Optional[np.ndarray | tuple[int, int, int]] = None,
	color_tolerance: float = 8.0,
	keep_largest_component_only: bool = False,
	fill_internal_holes: bool = False,
	remove_speckles: bool = False,
	min_speckle_neighbors: int = 3,
	speckle_iterations: int = 1,
) -> EdgeDetectionResult:
	fg_mask = build_foreground_mask(
		image_rgb=image_rgb,
		background_rgb=background_rgb,
		color_tolerance=color_tolerance,
	)
	if keep_largest_component_only:
		fg_mask = keep_largest_connected_component(fg_mask)
	if fill_internal_holes:
		fg_mask = _fill_internal_holes(fg_mask)
	if remove_speckles:
		fg_mask = _remove_speckles(
			fg_mask,
			min_neighbors=min_speckle_neighbors,
			iterations=speckle_iterations,
		)
		if keep_largest_component_only:
			fg_mask = keep_largest_connected_component(fg_mask)

	edge_mask = extract_edge_mask(fg_mask)
	coords = edge_coordinates_from_mask(edge_mask)

	return EdgeDetectionResult(
		edge_coordinates_xy=coords,
		edge_mask=edge_mask,
		foreground_mask=fg_mask,
	)


def find_body_edge_pixels_from_path(
	image_path: str | Path,
	background_rgb: Optional[np.ndarray | tuple[int, int, int]] = None,
	color_tolerance: float = 8.0,
	keep_largest_component_only: bool = False,
	fill_internal_holes: bool = False,
	remove_speckles: bool = False,
	min_speckle_neighbors: int = 3,
	speckle_iterations: int = 1,
) -> EdgeDetectionResult:
	image_rgb = load_image_rgb(image_path)
	return find_body_edge_pixels(
		image_rgb=image_rgb,
		background_rgb=background_rgb,
		color_tolerance=color_tolerance,
		keep_largest_component_only=keep_largest_component_only,
		fill_internal_holes=fill_internal_holes,
		remove_speckles=remove_speckles,
		min_speckle_neighbors=min_speckle_neighbors,
		speckle_iterations=speckle_iterations,
	)


if __name__ == "__main__":
	target = Path("Body_Images") / "celestial_render.png"
	if target.exists():
		result = find_body_edge_pixels_from_path(target)
		print(f"Detected edge pixels: {len(result.edge_coordinates_xy)}")
		print("First 10 coordinates (x, y):")
		print(result.edge_coordinates_xy[:10])
	else:
		print(f"No demo file at: {target}")
