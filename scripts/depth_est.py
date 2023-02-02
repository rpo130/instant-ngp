import os.path

from common import *

import json
import cv2

import pyngp as ngp


def depth_est(scene_config_file, depth_dir="depth", sigma_thrsh=15, snapshot_file="base.msgpack"):
	transforms_file = scene_config_file
	scene_dir = os.path.dirname(scene_config_file)
	depth_dir_abs = os.path.join(scene_dir, depth_dir)
	dex_depth_dir_abs = os.path.join(scene_dir, depth_dir+"_dex")

	if not os.path.exists(depth_dir_abs):
		os.mkdir(depth_dir_abs)
	if not os.path.exists(dex_depth_dir_abs):
		os.mkdir(dex_depth_dir_abs)

	poses = []
	img_names = []

	with open(transforms_file, 'r') as tf:
		meta = json.load(tf)
		for frame in meta['frames']:
			poses.append(frame['transform_matrix'])
			basename = os.path.basename(frame['file_path'])
			if not os.path.splitext(basename)[1]:
				basename = basename + ".png"
			img_names.append(basename)

	width = int(meta['w'])
	height = int(meta['h'])
	camera_angle_x = meta['camera_angle_x']
	camera_angle_y = meta['camera_angle_y']

	mode = ngp.TestbedMode.Nerf
	configs_dir = os.path.join(ROOT_DIR, "configs", "nerf")
	testbed = ngp.Testbed(mode)
	# testbed.nerf.sharpen = float(0)
	testbed.shall_train = False

	# Load a trained NeRF model
	print("Loading snapshot ", snapshot_file)
	testbed.load_snapshot(snapshot_file)
	testbed.nerf.render_with_camera_distortion = True
	testbed.snap_to_pixel_centers = True
	spp = 1
	testbed.nerf.rendering_min_transmittance = 1e-4
	testbed.fov_axis = 0
	testbed.fov = camera_angle_x * 180 / np.pi
	# testbed.fov_axis = 1
	# testbed.fov = camera_angle_y * 180 / np.pi

	# Set render mode
	testbed.render_mode = ngp.RenderMode.Depth

	# Adjust DeX threshold value
	testbed.dex_nerf = True
	testbed.sigma_thrsh = sigma_thrsh

	# Set camera matrix
	for img_name, c2w_matrix in zip(img_names, poses):
		testbed.set_nerf_camera_matrix(np.matrix(c2w_matrix)[:-1, :])
		# Render estimated depth
		print(f'rendering dex {img_name}')
		depth_raw = testbed.render(width, height, spp, True)  # raw depth values (float, in m)
		depth_raw = depth_raw[..., 0]
		depth_int = depth_raw
		depth_int = depth_int.astype(np.uint8)
		cv2.imwrite(os.path.join(dex_depth_dir_abs, img_name), depth_int)

	# Set camera matrix
	testbed.dex_nerf = False
	for img_name, c2w_matrix in zip(img_names, poses):
		testbed.set_nerf_camera_matrix(np.matrix(c2w_matrix)[:-1, :])
		print(f'rendering {img_name}')
		depth_raw = testbed.render(width, height, spp, True)  # raw depth values (float, in m)
		depth_raw = depth_raw[..., 0]
		depth_int = depth_raw
		depth_int = depth_int.astype(np.uint8)
		cv2.imwrite(os.path.join(depth_dir_abs, img_name), depth_int)

	ngp.free_temporary_memory()

if __name__ == "__main__":
	args = sys.argv[1:]
	path = args[0]
	snap = args[1]
	depth_est(scene_config_file=path, snapshot_file=snap)