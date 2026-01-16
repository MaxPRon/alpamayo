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

# Detailed inference script with per-stage timing information
# This provides granular timing breakdowns of the inference pipeline

import torch
import numpy as np
import time
from contextlib import contextmanager
from typing import Dict, Any
import json
from pathlib import Path
from datetime import datetime

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper


class InferenceProfiler:
    """Context manager for profiling inference stages."""
    
    def __init__(self, use_cuda: bool = True):
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.timings = {}
        
    @contextmanager
    def profile(self, name: str):
        """Profile a code block."""
        if self.use_cuda:
            torch.cuda.synchronize()
        start = time.time()
        
        try:
            yield
        finally:
            if self.use_cuda:
                torch.cuda.synchronize()
            elapsed = time.time() - start
            self.timings[name] = elapsed
            
    def get_timings(self) -> Dict[str, float]:
        """Get all recorded timings."""
        return self.timings.copy()
    
    def print_summary(self):
        """Print a formatted summary of timings."""
        print("\n" + "=" * 80)
        print("INFERENCE TIMING BREAKDOWN")
        print("=" * 80)
        total = sum(self.timings.values())
        for name, elapsed in self.timings.items():
            percentage = (elapsed / total * 100) if total > 0 else 0
            print(f"{name:.<50} {elapsed:>8.3f}s ({percentage:>5.1f}%)")
        print("-" * 80)
        print(f"{'TOTAL':.<50} {total:>8.3f}s (100.0%)")
        print("=" * 80)


def detailed_inference(
    clip_id: str,
    t0_us: int = 5_100_000,
    num_traj_samples: int = 1,
    top_p: float = 0.98,
    temperature: float = 0.6,
    save_results: bool = True,
):
    """Run inference with detailed timing breakdown.
    
    Args:
        clip_id: The clip ID to process
        t0_us: The start time in microseconds
        num_traj_samples: Number of trajectory samples
        top_p: Top-p sampling parameter
        temperature: Temperature for sampling
        save_results: Whether to save timing results to disk
    
    Returns:
        Dictionary containing predictions and timing information
    """
    profiler = InferenceProfiler()
    
    print(f"Processing clip: {clip_id}")
    print(f"Parameters: num_traj_samples={num_traj_samples}, top_p={top_p}, temperature={temperature}")
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("\nWARNING: CUDA not available, running on CPU!")
    
    # Load model
    print("\nLoading model...")
    with profiler.profile("model_loading"):
        model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16).to("cuda")
        processor = helper.get_processor(model.tokenizer)
    print(f"Model loaded in {profiler.timings['model_loading']:.3f}s")
    
    # Load data
    print("\nLoading dataset...")
    with profiler.profile("data_loading"):
        data = load_physical_aiavdataset(clip_id, t0_us=t0_us)
    num_images = data["image_frames"].shape[0] * data["image_frames"].shape[1]
    print(f"Dataset loaded in {profiler.timings['data_loading']:.3f}s")
    print(f"Number of images: {num_images}")
    print(f"Image shape: {data['image_frames'].shape}")
    
    # Prepare messages
    print("\nPreparing inputs...")
    with profiler.profile("message_creation"):
        messages = helper.create_message(data["image_frames"].flatten(0, 1))
    
    # Tokenization and preprocessing
    with profiler.profile("tokenization"):
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
    print(f"Sequence length: {inputs.input_ids.shape}")
    
    # Move to device
    with profiler.profile("data_transfer"):
        model_inputs = {
            "tokenized_data": inputs,
            "ego_history_xyz": data["ego_history_xyz"],
            "ego_history_rot": data["ego_history_rot"],
        }
        model_inputs = helper.to_device(model_inputs, "cuda")
    
    # Get GPU memory before inference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        mem_before = torch.cuda.memory_allocated() / 1024**3
        print(f"\nGPU memory before inference: {mem_before:.2f} GB")
    
    # Run inference
    print("\nRunning inference...")
    torch.cuda.manual_seed_all(42)
    
    with profiler.profile("inference_total"):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                top_p=top_p,
                temperature=temperature,
                num_traj_samples=num_traj_samples,
                max_generation_length=256,
                return_extra=True,
            )
    
    # Get GPU memory after inference
    if torch.cuda.is_available():
        mem_after = torch.cuda.memory_allocated() / 1024**3
        mem_peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"GPU memory after inference: {mem_after:.2f} GB")
        print(f"GPU memory peak: {mem_peak:.2f} GB")
        print(f"GPU memory increase: {mem_after - mem_before:.2f} GB")
    
    # Compute metrics
    with profiler.profile("metrics_computation"):
        gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
        pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
        diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
        min_ade = diff.min()
    
    print(f"\nminADE: {min_ade:.3f} meters")
    
    # Print timing summary
    profiler.print_summary()
    
    # Calculate derived metrics
    inference_time = profiler.timings["inference_total"]
    time_per_image = inference_time / num_images
    
    print("\nDERIVED METRICS")
    print("=" * 80)
    print(f"Time per image: {time_per_image:.3f}s")
    print(f"Images per second: {1/time_per_image:.2f}")
    print(f"Total processing time (excluding model load): {sum(profiler.timings.values()) - profiler.timings['model_loading']:.3f}s")
    print("=" * 80)
    
    # Prepare results
    results = {
        "clip_id": clip_id,
        "num_images": int(num_images),
        "num_traj_samples": num_traj_samples,
        "min_ade_meters": float(min_ade),
        "timings": profiler.get_timings(),
        "time_per_image_seconds": time_per_image,
        "images_per_second": 1 / time_per_image,
        "predictions": {
            "xyz_shape": list(pred_xyz.shape),
            "rot_shape": list(pred_rot.shape),
        },
        "chain_of_causation": [str(cot) for cot in (extra["cot"][0].tolist() if isinstance(extra["cot"][0], np.ndarray) else extra["cot"][0])],
        "timestamp": datetime.now().isoformat(),
    }
    
    if torch.cuda.is_available():
        results["gpu_info"] = {
            "device_name": torch.cuda.get_device_name(0),
            "memory_before_gb": mem_before,
            "memory_after_gb": mem_after,
            "memory_peak_gb": mem_peak,
            "memory_increase_gb": mem_after - mem_before,
        }
    
    # Save results
    if save_results:
        output_dir = Path("inference_results") / "detailed" / datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "timing_breakdown.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_dir}")
    
    return results


def main():
    """Main function."""
    # Example usage
    clip_id = "030c760c-ae38-49aa-9ad8-f5650a545d26"
    
    results = detailed_inference(
        clip_id=clip_id,
        num_traj_samples=1,  # Increase for more trajectory samples
        top_p=0.98,
        temperature=0.6,
        save_results=True,
    )
    
    print("\n" + "=" * 80)
    print("NOTES")
    print("=" * 80)
    print("- Images are processed as a temporal sequence (typically 4 frames)")
    print("- The model uses multi-frame context for trajectory prediction")
    print("- Time per image = Total inference time / Number of images")
    print("- Individual image processing is not possible due to temporal dependencies")
    print("=" * 80)


if __name__ == "__main__":
    main()
