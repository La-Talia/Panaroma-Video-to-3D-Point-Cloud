#!/usr/bin/env python3
"""
2D-to-3D Monocular Panoramic Point Cloud Reconstruction Pipeline
================================================================
Converts a short hand-held panoramic indoor room video into a dense 3D colored point cloud (.ply).
Supports both ZoeDepth (metric depth in meters) and Depth Anything V2 (relative depth).

Usage Examples:
    python pipeline.py --video sample.mp4 --model zoedepth --output room_zoedepth.ply
    python pipeline.py --video sample.mp4 --model depth_anything --output room_depth_anything.ply
    python pipeline.py --image panorama.jpg --model zoedepth --stride 1
"""

import argparse
import os
import sys
import glob
import numpy as np
import cv2
from PIL import Image
from scipy.spatial import cKDTree
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert indoor panoramic video to dense 3D colored point cloud."
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--video", type=str, default="sample.mp4", help="Path to input video (default: sample.mp4)")
    group.add_argument("--image", type=str, default=None, help="Path to pre-stitched panorama image")
    
    parser.add_argument(
        "--model",
        type=str,
        choices=["zoedepth", "depth_anything"],
        default="zoedepth",
        help="Depth estimation model to use: 'zoedepth' (metric) or 'depth_anything' (relative)"
    )
    parser.add_argument("--fps", type=float, default=2.0, help="Frame extraction rate (default: 2 fps)")
    parser.add_argument("--fov_x", type=float, default=None, help="Horizontal FOV in degrees (default: 120 for ZoeDepth, 140 for DepthAnything)")
    parser.add_argument("--stride", type=int, default=2, help="Point cloud sampling stride (default: 2)")
    parser.add_argument("--output", type=str, default=None, help="Output .ply file path")
    parser.add_argument("--preview", action="store_true", help="Launch interactive 3D preview in browser")
    parser.add_argument("--save_html", type=str, default=None, help="Save interactive 3D HTML visualization to path")
    parser.add_argument("--device", type=str, default=None, help="Torch device ('cuda', 'cpu', 'mps')")
    
    return parser.parse_args()


def extract_frames(video_path: str, fps: float = 2.0):
    """Extract frames from video at the specified fps using OpenCV."""
    print(f"[Stage 1/6] Extracting frames from {video_path} at {fps} fps...")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(round(video_fps / fps)))
    
    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            frames.append(frame)
        frame_idx += 1
    
    cap.release()
    print(f"            Extracted {len(frames)} frames.")
    if len(frames) < 2:
        raise ValueError("Need at least 2 frames for panoramic stitching.")
    return frames


def stitch_panorama(frames):
    """Stitch a list of frames into a single wide panorama and crop borders."""
    print("[Stage 2/6] Stitching frames into a panoramic mosaic...")
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    status, stitched = stitcher.stitch(frames)
    
    if status != cv2.Stitcher_OK:
        # Fallback to SCANS mode if PANORAMA fails
        print("            Standard stitch failed, retrying in SCANS mode...")
        stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
        status, stitched = stitcher.stitch(frames)
        if status != cv2.Stitcher_OK:
            raise RuntimeError(f"Panoramic stitching failed with error code {status}. Ensure frames have sufficient overlap.")

    # Crop black irregular borders from homography warping
    print("            Cropping stitching boundary borders...")
    gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        pad_x, pad_y = int(w * 0.04), int(h * 0.04)
        stitched = stitched[y + pad_y : y + h - pad_y, x + pad_x : x + w - pad_x]

    rgb_pano = cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB)
    pil_pano = Image.fromarray(rgb_pano)
    print(f"            Panorama resolution: {pil_pano.width} x {pil_pano.height}")
    return pil_pano


def estimate_depth(pil_img, model_type: str, device: str):
    """Estimate depth map using ZoeDepth or Depth Anything V2."""
    print(f"[Stage 3/6] Running depth estimation with model '{model_type}' on {device}...")
    import torch
    
    if model_type == "zoedepth":
        # Load ZoeDepth metric model fine-tuned on NYUv2
        zoe = torch.hub.load("isl-org/ZoeDepth", "ZoeD_N", pretrained=True, trust_repo=True).to(device).eval()
        with torch.no_grad():
            depth_map = zoe.infer_pil(pil_img)
        # Bilateral filter for noise smoothing while preserving edges
        depth_map = cv2.bilateralFilter(depth_map.astype(np.float32), d=9, sigmaColor=0.5, sigmaSpace=15)
        return depth_map
        
    elif model_type == "depth_anything":
        from transformers import pipeline
        pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=device)
        res = pipe(pil_img)
        raw_tensor = res["predicted_depth"].squeeze().cpu().numpy()
        width, height = pil_img.size
        disparity = cv2.resize(raw_tensor, (width, height), interpolation=cv2.INTER_CUBIC)
        
        # Invert disparity to obtain depth proxy
        depth_raw = 1.0 / (disparity + 1e-5)
        p5, p95 = np.percentile(depth_raw, 5), np.percentile(depth_raw, 95)
        depth_clipped = np.clip(depth_raw, p5, p95)
        depth_norm = (depth_clipped - p5) / (p95 - p5 + 1e-5)
        depth_scaled = (depth_norm * 5.0) + 1.0  # approximate 1 to 6 unit room scale
        depth_map = cv2.bilateralFilter(depth_scaled.astype(np.float32), d=9, sigmaColor=0.1, sigmaSpace=10)
        return depth_map
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def backproject_to_3d(depth_map, pano_img, model_type: str, fov_x_deg: float, stride: int = 2):
    """Back-project 2D depth and RGB pixels into 3D Cartesian coordinates."""
    print(f"[Stage 4/6] Back-projecting pixels to 3D (stride={stride}, FOV_X={fov_x_deg}°)...")
    width, height = pano_img.size
    
    u, v = np.meshgrid(np.arange(0, width, stride), np.arange(0, height, stride))
    depth_sampled = depth_map[::stride, ::stride]
    color_sampled = np.array(pano_img)[::stride, ::stride]
    
    fov_x_rad = np.deg2rad(fov_x_deg)
    
    if model_type == "zoedepth":
        # Spherical back-projection (metric depth in meters)
        fov_y_rad = fov_x_rad * (height / width)
        theta = (u / width - 0.5) * fov_x_rad
        phi = (v / height - 0.5) * fov_y_rad
        
        X = depth_sampled * np.sin(theta) * np.cos(phi)
        Y = -depth_sampled * np.sin(phi)
        Z = depth_sampled * np.cos(theta) * np.cos(phi)
    else:
        # Cylindrical back-projection (heuristic relative depth)
        theta = (u / width - 0.5) * fov_x_rad
        focal_length = width / fov_x_rad
        
        X = depth_sampled * np.sin(theta)
        Z = depth_sampled * np.cos(theta)
        Y = -(v - height / 2.0) * (depth_sampled / focal_length)
        
    points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
    colors = (color_sampled / 255.0).reshape(-1, 3)
    print(f"            Initial points generated: {len(points):,}")
    return points, colors


def remove_statistical_outliers(points, colors, k: int = 31, std_ratio: float = 1.5):
    """Remove noise using KDTree nearest neighbor distance thresholding."""
    print(f"[Stage 5/6] Removing statistical outliers (k={k}, std_ratio={std_ratio})...")
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=k)
    mean_dists = dists[:, 1:].mean(axis=1)
    
    threshold = mean_dists.mean() + std_ratio * mean_dists.std()
    mask = mean_dists < threshold
    
    filtered_points = points[mask]
    filtered_colors = colors[mask]
    print(f"            Retained {len(filtered_points):,} / {len(points):,} points ({len(filtered_points)/len(points)*100:.1f}%)")
    return filtered_points, filtered_colors


def save_ply(points, colors, output_path: str):
    """Export 3D points and colors to binary or ASCII PLY format."""
    print(f"[Stage 6/6] Saving point cloud to {output_path}...")
    try:
        from plyfile import PlyData, PlyElement
        vertex_colors = (colors * 255).astype(np.uint8)
        vertex_data = np.zeros(len(points), dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
        ])
        vertex_data['x'], vertex_data['y'], vertex_data['z'] = points[:, 0], points[:, 1], points[:, 2]
        vertex_data['red'], vertex_data['green'], vertex_data['blue'] = vertex_colors[:, 0], vertex_colors[:, 1], vertex_colors[:, 2]
        el = PlyElement.describe(vertex_data, 'vertex')
        PlyData([el], text=False).write(output_path)
    except ImportError:
        # Fallback to ASCII PLY writer without extra dependencies
        colors_255 = (colors * 255).astype(np.uint8)
        vertex_data = np.hstack((points, colors_255))
        header = (
            "ply\n"
            "format ascii 1.0\n"
            f"element vertex {vertex_data.shape[0]}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )
        with open(output_path, "w") as f:
            f.write(header)
            np.savetxt(f, vertex_data, fmt="%.4f %.4f %.4f %d %d %d")
            
    print(f"[Success] Export complete: {output_path} ({os.path.getsize(output_path):,} bytes)")


def visualize_plotly(points, colors, max_points: int = 40000, show: bool = True, html_path: str = None):
    """Render an interactive 3D point cloud scatter plot with Plotly."""
    import plotly.graph_objects as go
    
    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        p, c = points[idx], colors[idx]
    else:
        p, c = points, colors
        
    color_strings = [f"rgb({int(r*255)},{int(g*255)},{int(b*255)})" for r, g, b in c]
    fig = go.Figure(data=[go.Scatter3d(
        x=p[:, 0], y=p[:, 1], z=p[:, 2],
        mode="markers",
        marker=dict(size=1.5, color=color_strings)
    )])
    fig.update_layout(
        title="Indoor Panoramic 3D Point Cloud",
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, b=0, t=30),
        height=750,
    )
    if html_path:
        fig.write_html(html_path)
        print(f"[Visualization] Saved interactive 3D HTML to {html_path}")
    if show:
        fig.show()


def main():
    args = parse_args()
    import torch
    
    # Auto-detect device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Default FOV based on model
    fov_x = args.fov_x if args.fov_x is not None else (120.0 if args.model == "zoedepth" else 140.0)
    
    # Default output path
    output_path = args.output or f"room_{args.model}.ply"

    # 1. Obtain panorama
    if args.image:
        if not os.path.exists(args.image):
            raise FileNotFoundError(f"Image not found: {args.image}")
        pano_img = Image.open(args.image).convert("RGB")
        print(f"[Input] Using provided panorama image: {args.image}")
    else:
        video_path = args.video
        frames = extract_frames(video_path, fps=args.fps)
        pano_img = stitch_panorama(frames)

    # 2. Depth estimation
    depth_map = estimate_depth(pano_img, model_type=args.model, device=device)

    # 3. 3D Back-projection
    points, colors = backproject_to_3d(depth_map, pano_img, model_type=args.model, fov_x_deg=fov_x, stride=args.stride)

    # 4. Outlier removal
    k = 31 if args.model == "zoedepth" else 21
    std_ratio = 1.5 if args.model == "zoedepth" else 2.0
    filtered_points, filtered_colors = remove_statistical_outliers(points, colors, k=k, std_ratio=std_ratio)

    # 5. PLY export
    save_ply(filtered_points, filtered_colors, output_path)

    # 6. Interactive visualization
    if args.preview or args.save_html:
        visualize_plotly(filtered_points, filtered_colors, show=args.preview, html_path=args.save_html)


if __name__ == "__main__":
    main()
