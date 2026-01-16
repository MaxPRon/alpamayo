# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Batch inference script that processes multiple images, saves results as
# individual images and videos, and measures inference time.

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
from datetime import datetime
import json
import subprocess

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Warning: OpenCV not available. Video saving will be disabled.")

try:
    import mediapy as mp
    HAS_MEDIAPY = True
except ImportError:
    HAS_MEDIAPY = False
    print("Warning: mediapy not available. Some visualization features may be limited.")


def get_gpu_info():
    """Get GPU information and utilization."""
    if not torch.cuda.is_available():
        return {"available": False, "message": "CUDA not available"}
    
    gpu_info = {
        "available": True,
        "device_name": torch.cuda.get_device_name(0),
        "device_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        "memory_allocated_gb": torch.cuda.memory_allocated(0) / 1024**3,
        "memory_reserved_gb": torch.cuda.memory_reserved(0) / 1024**3,
    }
    
    # Try to get GPU utilization using nvidia-smi
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', 
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            output = result.stdout.strip().split('\n')[0].split(',')
            gpu_info["gpu_utilization_percent"] = float(output[0].strip())
            gpu_info["memory_used_mb"] = float(output[1].strip())
            gpu_info["memory_total_mb"] = float(output[2].strip())
    except Exception as e:
        gpu_info["nvidia_smi_error"] = str(e)
    
    return gpu_info


def rotate_90cc(xy):
    """Rotate (x, y) by 90 deg CCW -> (y, -x)"""
    return np.stack([-xy[1], xy[0]], axis=0)


def save_trajectory_plot(pred_xyz, gt_xyz, output_path, clip_id, inference_time, min_ade):
    """Save a trajectory plot comparing predictions and ground truth."""
    plt.figure(figsize=(10, 8))
    
    # Plot predicted trajectories
    for i in range(pred_xyz.shape[2]):
        pred_xy = pred_xyz.cpu()[0, 0, i, :, :2].T.numpy()
        pred_xy_rot = rotate_90cc(pred_xy)
        plt.plot(*pred_xy_rot, "o-", label=f"Predicted Trajectory #{i + 1}", linewidth=2)
    
    # Plot ground truth
    gt_xy = gt_xyz.cpu()[0, 0, :, :2].T.numpy()
    gt_xy_rot = rotate_90cc(gt_xy)
    plt.plot(*gt_xy_rot, "r-", label="Ground Truth Trajectory", linewidth=2)
    
    plt.ylabel("y coordinate (meters)", fontsize=12)
    plt.xlabel("x coordinate (meters)", fontsize=12)
    plt.title(f"Clip: {clip_id}\nInference Time: {inference_time:.3f}s | minADE: {min_ade:.3f}m", 
              fontsize=10)
    plt.legend(loc="best")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved trajectory plot: {output_path}")


def save_input_images(image_frames, output_path):
    """Save input images as a grid."""
    images = image_frames.flatten(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    
    n_images = images.shape[0]
    cols = min(4, n_images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col] if rows > 1 or cols > 1 else axes
        
        # Normalize image for display
        img_display = img
        if img_display.max() > 1.0:
            img_display = img_display / 255.0
        
        ax.imshow(img_display)
        ax.axis('off')
        ax.set_title(f"Frame {idx + 1}", fontsize=10)
    
    # Hide unused subplots
    for idx in range(n_images, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved input images: {output_path}")


def create_video_with_trajectory(image_frames, pred_xyz, gt_xyz, output_path, fps=10):
    """Create a video showing input frames with trajectory overlay."""
    if not HAS_CV2:
        print("  Skipping video creation (OpenCV not available)")
        return
    
    images = image_frames.flatten(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    
    # Normalize images
    if images.max() > 1.0:
        images = images / 255.0
    images = (images * 255).astype(np.uint8)
    
    # Get dimensions
    h, w = images.shape[1:3]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    
    # Write frames
    for img in images:
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        video_writer.write(img_bgr)
    
    video_writer.release()
    print(f"  Saved video: {output_path}")


def process_clip(model, processor, clip_id, output_dir, t0_us=5_100_000, 
                 num_traj_samples=1, top_p=0.98, temperature=0.6):
    """Process a single clip and save results."""
    print(f"\nProcessing clip: {clip_id}")
    
    # Create output directory for this clip
    clip_output_dir = output_dir / clip_id
    clip_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get GPU info before processing
    gpu_info_before = get_gpu_info()
    
    # Load data
    print("  Loading dataset...")
    load_start = time.time()
    data = load_physical_aiavdataset(clip_id, t0_us=t0_us)
    load_time = time.time() - load_start
    print(f"  Dataset loaded in {load_time:.3f}s")
    
    # Count number of images
    num_images = data["image_frames"].shape[0] * data["image_frames"].shape[1]
    print(f"  Number of images: {num_images}")
    
    # Save input images
    save_input_images(data["image_frames"], clip_output_dir / "input_images.png")
    
    # Prepare inputs
    messages = helper.create_message(data["image_frames"].flatten(0, 1))
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    model_inputs = helper.to_device(model_inputs, "cuda")
    
    # Run inference with timing
    print("  Running inference...")
    torch.cuda.synchronize()
    inference_start = time.time()
    
    torch.cuda.manual_seed_all(42)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=top_p,
            temperature=temperature,
            num_traj_samples=num_traj_samples,
            max_generation_length=256,
            return_extra=True,
        )
    
    torch.cuda.synchronize()
    inference_time = time.time() - inference_start
    print(f"  Inference completed in {inference_time:.3f}s")
    print(f"  Average time per image: {inference_time / num_images:.3f}s")
    
    # Get GPU info after processing
    gpu_info_after = get_gpu_info()
    
    # Compute metrics
    gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
    diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
    min_ade = diff.min()
    print(f"  minADE: {min_ade:.3f} meters")
    
    # Save trajectory plot
    save_trajectory_plot(
        pred_xyz, 
        data["ego_future_xyz"], 
        clip_output_dir / "trajectory_plot.png",
        clip_id,
        inference_time,
        min_ade
    )
    
    # Save video
    create_video_with_trajectory(
        data["image_frames"],
        pred_xyz,
        data["ego_future_xyz"],
        clip_output_dir / "input_video.mp4"
    )
    
    # Save Chain-of-Causation reasoning
    cot_text = extra["cot"][0]
    # Convert to list if it's a numpy array
    if isinstance(cot_text, np.ndarray):
        cot_text = cot_text.tolist()
    elif not isinstance(cot_text, list):
        cot_text = [cot_text]
    
    with open(clip_output_dir / "chain_of_causation.txt", "w") as f:
        for idx, cot in enumerate(cot_text):
            f.write(f"Trajectory #{idx + 1}:\n")
            f.write(str(cot) + "\n")
            f.write("-" * 80 + "\n\n")
    print(f"  Saved Chain-of-Causation reasoning")
    
    # Save metrics
    metrics = {
        "clip_id": clip_id,
        "num_images": int(num_images),
        "load_time_seconds": load_time,
        "inference_time_seconds": inference_time,
        "time_per_image_seconds": inference_time / num_images,
        "min_ade_meters": float(min_ade),
        "num_traj_samples": num_traj_samples,
        "top_p": top_p,
        "temperature": temperature,
        "gpu_info_before": gpu_info_before,
        "gpu_info_after": gpu_info_after,
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(clip_output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics")
    
    return metrics


def main():
    """Main function to process multiple clips."""
    # Configuration
    clip_ids = [
        "030c760c-ae38-49aa-9ad8-f5650a545d26",  # Default clip
        # Add more clip IDs here
    ]
    
    output_base_dir = Path("inference_results")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_base_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Processing {len(clip_ids)} clips...")
    
    # Check GPU availability
    print("\n" + "=" * 80)
    print("GPU INFORMATION")
    print("=" * 80)
    initial_gpu_info = get_gpu_info()
    if initial_gpu_info["available"]:
        print(f"GPU Available: Yes")
        print(f"Device: {initial_gpu_info['device_name']}")
        print(f"Device Count: {initial_gpu_info['device_count']}")
        print(f"Current Device: {initial_gpu_info['current_device']}")
        print(f"Memory Allocated: {initial_gpu_info['memory_allocated_gb']:.2f} GB")
        print(f"Memory Reserved: {initial_gpu_info['memory_reserved_gb']:.2f} GB")
        if "gpu_utilization_percent" in initial_gpu_info:
            print(f"GPU Utilization: {initial_gpu_info['gpu_utilization_percent']:.1f}%")
        if "memory_used_mb" in initial_gpu_info and "memory_total_mb" in initial_gpu_info:
            print(f"Memory Used: {initial_gpu_info['memory_used_mb']:.0f} MB / {initial_gpu_info['memory_total_mb']:.0f} MB")
    else:
        print(f"GPU Available: No - {initial_gpu_info['message']}")
        print("WARNING: Running on CPU will be very slow!")
    print("=" * 80)
    
    # Load model
    print("\nLoading model...")
    model_load_start = time.time()
    model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16).to("cuda")
    processor = helper.get_processor(model.tokenizer)
    model_load_time = time.time() - model_load_start
    print(f"Model loaded in {model_load_time:.3f}s")
    
    # Process each clip
    all_metrics = []
    total_start = time.time()
    
    for clip_id in clip_ids:
        try:
            metrics = process_clip(
                model, 
                processor, 
                clip_id, 
                output_dir,
                num_traj_samples=1,  # Increase for more trajectory samples
            )
            all_metrics.append(metrics)
        except Exception as e:
            print(f"  Error processing clip {clip_id}: {e}")
            import traceback
            traceback.print_exc()
    
    total_time = time.time() - total_start
    
    # Save summary
    summary = {
        "total_clips": len(clip_ids),
        "successful_clips": len(all_metrics),
        "model_load_time_seconds": model_load_time,
        "total_processing_time_seconds": total_time,
        "average_inference_time_seconds": np.mean([m["inference_time_seconds"] for m in all_metrics]) if all_metrics else 0,
        "average_time_per_image_seconds": np.mean([m["time_per_image_seconds"] for m in all_metrics]) if all_metrics else 0,
        "average_min_ade_meters": np.mean([m["min_ade_meters"] for m in all_metrics]) if all_metrics else 0,
        "total_images_processed": sum([m["num_images"] for m in all_metrics]) if all_metrics else 0,
        "gpu_info": initial_gpu_info,
        "metrics_per_clip": all_metrics,
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total clips processed: {len(all_metrics)}/{len(clip_ids)}")
    print(f"Model load time: {model_load_time:.3f}s")
    print(f"Total processing time: {total_time:.3f}s")
    if all_metrics:
        print(f"Average inference time per clip: {summary['average_inference_time_seconds']:.3f}s")
        print(f"Average time per image: {summary['average_time_per_image_seconds']:.3f}s")
        print(f"Total images processed: {summary['total_images_processed']}")
        print(f"Average minADE: {summary['average_min_ade_meters']:.3f}m")
    print(f"\nResults saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
