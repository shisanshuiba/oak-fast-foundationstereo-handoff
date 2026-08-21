# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os, sys

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')

from omegaconf import OmegaConf
from core.utils.utils import InputPadder
import argparse, torch, imageio, logging, yaml
import numpy as np
from Utils import (
    AMP_DTYPE, set_logging_format, set_seed, vis_disparity,
    depth2xyzmap, toOpen3dCloud, o3d,
)
import cv2


def depth_to_pseudocolor(depth, valid_mask=None, min_depth=None, max_depth=None):
  """Map metric depth to RGB Turbo colors (near=red, far=blue)."""
  depth = np.asarray(depth, dtype=np.float32)
  if depth.ndim != 2:
    raise ValueError(f'depth must be a 2D array, got shape {depth.shape}')

  valid = np.isfinite(depth) & (depth > 0)
  if valid_mask is not None:
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if valid_mask.shape != depth.shape:
      raise ValueError(
          f'valid_mask shape {valid_mask.shape} does not match depth shape {depth.shape}'
      )
    valid &= valid_mask

  if not valid.any():
    raise ValueError('No valid depth pixels are available for pseudo-color mapping')

  valid_depth = depth[valid]
  color_min = float(np.percentile(valid_depth, 2)) if min_depth is None else float(min_depth)
  color_max = float(np.percentile(valid_depth, 98)) if max_depth is None else float(max_depth)

  if not np.isfinite(color_min) or not np.isfinite(color_max):
    raise ValueError('Pseudo-color depth bounds must be finite')
  if color_max <= color_min:
    if min_depth is not None or max_depth is not None:
      raise ValueError(
          f'color_max_depth ({color_max}) must be greater than '
          f'color_min_depth ({color_min})'
      )
    color_max = color_min + max(abs(color_min) * 1e-6, 1e-6)

  normalized = np.zeros(depth.shape, dtype=np.float32)
  normalized[valid] = np.clip(
      (color_max - depth[valid]) / (color_max - color_min), 0.0, 1.0
  )
  color_index = np.round(normalized * 255).astype(np.uint8)
  color_bgr = cv2.applyColorMap(color_index, cv2.COLORMAP_TURBO)
  color_rgb = color_bgr[..., ::-1].copy()
  color_rgb[~valid] = 0
  return color_rgb, color_min, color_max


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_dir', default=f'{code_dir}/../weights/23-36-37/model_best_bp2_serialize.pth', type=str)
  parser.add_argument('--left_file', default=f'{code_dir}/../demo_data/left.png', type=str)
  parser.add_argument('--right_file', default=f'{code_dir}/../demo_data/right.png', type=str)
  parser.add_argument('--intrinsic_file', default=f'{code_dir}/../demo_data/K.txt', type=str, help='camera intrinsic matrix and baseline file')
  parser.add_argument('--out_dir', default=f'{code_dir}/../output/pseudocolor', type=str)
  parser.add_argument('--remove_invisible', default=1, type=int)
  parser.add_argument('--denoise_cloud', default=0, type=int)
  parser.add_argument('--denoise_nb_points', type=int, default=30, help='number of points to consider for radius outlier removal')
  parser.add_argument('--denoise_radius', type=float, default=0.03, help='radius to use for outlier removal')
  parser.add_argument('--scale', default=1, type=float)
  parser.add_argument('--hiera', default=0, type=int)
  parser.add_argument('--get_pc', type=int, default=1, help='save point cloud output')
  parser.add_argument('--valid_iters', type=int, default=8, help='number of flow-field updates during forward pass')
  parser.add_argument('--max_disp', type=int, default=192, help='maximum disparity')
  parser.add_argument('--zfar', type=float, default=100, help='max depth to include in point cloud')
  parser.add_argument('--color_min_depth', type=float, default=None, help='depth in meters mapped to red; defaults to the valid-depth 2nd percentile')
  parser.add_argument('--color_max_depth', type=float, default=None, help='depth in meters mapped to blue; defaults to the valid-depth 98th percentile')
  parser.add_argument('--show_windows', type=int, default=1, help='show disparity and Open3D windows (0: no, 1: yes)')
  cli_args = parser.parse_args()

  set_logging_format()
  set_seed(0)
  torch.autograd.set_grad_enabled(False)

  os.makedirs(cli_args.out_dir, exist_ok=True)

  with open(f'{os.path.dirname(cli_args.model_dir)}/cfg.yaml', 'r') as ff:
    cfg: dict = yaml.safe_load(ff)
  for key, value in cli_args.__dict__.items():
    if value is not None:
      cfg[key] = value
  args = OmegaConf.create(cfg)
  logging.info(f'args:\n{args}')

  model = torch.load(args.model_dir, map_location='cpu', weights_only=False)
  model.args.valid_iters = args.valid_iters
  model.args.max_disp = args.max_disp
  model.cuda().eval()

  scale = args.scale
  img0 = imageio.imread(args.left_file)
  img1 = imageio.imread(args.right_file)
  if len(img0.shape) == 2:
    img0 = np.tile(img0[..., None], (1, 1, 3))
    img1 = np.tile(img1[..., None], (1, 1, 3))
  img0 = img0[..., :3]
  img1 = img1[..., :3]

  img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None)
  img1 = cv2.resize(img1, dsize=(img0.shape[1], img0.shape[0]))
  H, W = img0.shape[:2]
  img0_ori = img0.copy()
  img1_ori = img1.copy()
  logging.info(f'img0: {img0.shape}')
  imageio.imwrite(os.path.join(args.out_dir, 'left.png'), img0)
  imageio.imwrite(os.path.join(args.out_dir, 'right.png'), img1)

  img0 = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
  img1 = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)
  padder = InputPadder(img0.shape, divis_by=32, force_square=False)
  img0, img1 = padder.pad(img0, img1)

  logging.info('Start forward, 1st time run can be slow due to compilation')
  with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
    if not args.hiera:
      disp = model.forward(
          img0, img1, iters=args.valid_iters, test_mode=True,
          optimize_build_volume='pytorch1'
      )
    else:
      disp = model.run_hierachical(
          img0, img1, iters=args.valid_iters, test_mode=True, small_ratio=0.5
      )
  logging.info('forward done')
  disp = padder.unpad(disp.float())
  disp = disp.data.cpu().numpy().reshape(H, W).clip(0, None)

  disp_color = vis_disparity(
      disp, min_val=None, max_val=None, cmap=None,
      color_map=cv2.COLORMAP_TURBO
  )
  disp_vis = np.concatenate([img0_ori, img1_ori, disp_color], axis=1)
  imageio.imwrite(os.path.join(args.out_dir, 'disp_vis.png'), disp_vis)
  if args.show_windows:
    display_scale = 1280 / disp_vis.shape[1]
    resized_vis = cv2.resize(
        disp_vis,
        (int(disp_vis.shape[1] * display_scale), int(disp_vis.shape[0] * display_scale)),
    )
    cv2.imshow('disp', resized_vis[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyWindow('disp')

  if args.remove_invisible:
    yy, xx = np.meshgrid(
        np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing='ij'
    )
    us_right = xx - disp
    disp[us_right < 0] = np.inf

  if args.get_pc:
    if o3d is None:
      raise RuntimeError('open3d is required when --get_pc 1')

    with open(args.intrinsic_file, 'r') as ff:
      lines = ff.readlines()
      K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3, 3)
      baseline = float(lines[1])
    K[:2] *= scale

    with np.errstate(divide='ignore', invalid='ignore'):
      depth = K[0, 0] * baseline / disp
    np.save(os.path.join(args.out_dir, 'depth_meter.npy'), depth)

    xyz_map = depth2xyzmap(depth, K)
    point_depth = xyz_map[..., 2]
    keep_mask = (
        np.isfinite(point_depth) & (point_depth > 0) & (point_depth <= args.zfar)
    )
    keep_ids = np.flatnonzero(keep_mask.reshape(-1))
    if keep_ids.size == 0:
      raise RuntimeError('No valid 3D points remain after applying the depth filters')

    pseudo_rgb, color_min, color_max = depth_to_pseudocolor(
        depth,
        valid_mask=keep_mask,
        min_depth=cli_args.color_min_depth,
        max_depth=cli_args.color_max_depth,
    )
    imageio.imwrite(
        os.path.join(args.out_dir, 'depth_pseudocolor.png'), pseudo_rgb
    )
    logging.info(
        f'Pseudo-color depth range: {color_min:.4f} m (red/near) to '
        f'{color_max:.4f} m (blue/far)'
    )

    texture_pcd = toOpen3dCloud(
        xyz_map.reshape(-1, 3), img0_ori.reshape(-1, 3)
    ).select_by_index(keep_ids)
    pseudo_pcd = toOpen3dCloud(
        xyz_map.reshape(-1, 3), pseudo_rgb.reshape(-1, 3)
    ).select_by_index(keep_ids)

    texture_path = os.path.join(args.out_dir, 'cloud.ply')
    pseudo_path = os.path.join(args.out_dir, 'cloud_pseudocolor.ply')
    if not o3d.io.write_point_cloud(texture_path, texture_pcd):
      raise RuntimeError(f'Failed to write point cloud: {texture_path}')
    if not o3d.io.write_point_cloud(pseudo_path, pseudo_pcd):
      raise RuntimeError(f'Failed to write point cloud: {pseudo_path}')
    logging.info(f'Texture point cloud saved to {texture_path}')
    logging.info(f'Pseudo-color point cloud saved to {pseudo_path}')

    display_pcd = pseudo_pcd
    if args.denoise_cloud:
      logging.info('[Optional step] denoise pseudo-color point cloud...')
      downsampled = pseudo_pcd.voxel_down_sample(voxel_size=0.001)
      _, inlier_ids = downsampled.remove_radius_outlier(
          nb_points=args.denoise_nb_points, radius=args.denoise_radius
      )
      denoised_pcd = downsampled.select_by_index(inlier_ids)
      if len(denoised_pcd.points) == 0:
        logging.warning(
            'Denoising removed every point; keeping the original pseudo-color '
            'point cloud for visualization'
        )
      else:
        display_pcd = denoised_pcd
        denoised_path = os.path.join(
            args.out_dir, 'cloud_pseudocolor_denoise.ply'
        )
        if not o3d.io.write_point_cloud(denoised_path, display_pcd):
          raise RuntimeError(f'Failed to write point cloud: {denoised_path}')
        logging.info(
            f'Denoised pseudo-color point cloud saved to {denoised_path}'
        )

    if args.show_windows:
      logging.info('Visualizing pseudo-color point cloud. Press ESC to exit.')
      visualizer = o3d.visualization.Visualizer()
      visualizer.create_window(window_name='Fast-FoundationStereo Pseudo-color')
      visualizer.add_geometry(display_pcd)
      visualizer.get_render_option().point_size = 1.0
      visualizer.get_render_option().background_color = np.array([0.5, 0.5, 0.5])
      view_control = visualizer.get_view_control()
      view_control.set_front([0, 0, -1])
      nearest_id = np.asarray(display_pcd.points)[:, 2].argmin()
      view_control.set_lookat(np.asarray(display_pcd.points)[nearest_id])
      view_control.set_up([0, -1, 0])
      visualizer.run()
      visualizer.destroy_window()
