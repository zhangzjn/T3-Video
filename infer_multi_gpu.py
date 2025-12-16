#!/usr/bin/env python3
import os
import sys
import math
import json
import glob
import subprocess
import argparse
from PIL import Image
import imageio
import torch
from diffsynth import save_video, VideoData, load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
import numpy as np

"""
- 使用方法举例（在外层 shell 中运行一次 launcher）：
    python run_wan_infer.py --num_groups 8 --start_id 0 --end_id -1 ...
  或者直接启动某个组（子进程方式）：
    CUDA_VISIBLE_DEVICES=3 python run_wan_infer.py --group_id 3 --num_groups 8 --start_id 0 --end_id -1 ...
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', default='Wan-AI/Wan2.1-T2V-1.3B')
    parser.add_argument('--text_path', default='/z_vtzhang/pretrained/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth')
    parser.add_argument('--dit_path', default='/z_vtzhang/pretrained/Wan2.1-T2V-1.3B/diffusion_pytorch_model*.safetensors')
    parser.add_argument('--vae_path', default='/z_vtzhang/pretrained/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth')
    parser.add_argument('--img_path', default=None)
    parser.add_argument('--local_model_path', default='models')
    parser.add_argument('--dit_path_full_pretrained', default=None)
    parser.add_argument('--dit_path_lora', default=None)
    # parser.add_argument('--dit_path_lora', nargs='*', default=None)
    # parser.add_argument('--mode', default='full')
    parser.add_argument('--lora_alpha', type=float, default=1.0)
    parser.add_argument('--height', type=int, default=2176)
    parser.add_argument('--width', type=int, default=3840)
    parser.add_argument('--num_frames', type=int, default=81)
    parser.add_argument('--out_dir', default='output/T3-Video-Wan2.1-T2V-1.3B-4K-Vbench-seed0')
    parser.add_argument('--extra_inputs', default='')
    parser.add_argument('--json_file', type=str, default='4K-VBench/4K-Vbench.json')
    parser.add_argument('--data_root', type=str, default='4K-Vbench')
    parser.add_argument('--tiled', type=str, default='false')
    parser.add_argument('--cfg_scale', type=float, default=5.0)
    parser.add_argument('--cfg_merge', type=str, default='false')
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--ratio', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    # 新增参数
    parser.add_argument('--start_id', type=int, default=0, help='Starting sample index (inclusive), default 0')
    parser.add_argument('--end_id', type=int, default=-1, help='Ending sample index (exclusive), -1 means all samples')
    parser.add_argument('--num_groups', type=int, default=8, help='Number of parallel groups (usually equals number of GPUs)')
    parser.add_argument('--group_id', type=int, default=-1, help='Sub-task index (0..num_groups-1), default -1 means launcher mode which will auto-spawn subprocesses')
    return parser.parse_args()

def load_data_for_path(file_path):
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.svg'}
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.mpeg', '.mpg', '.webm'}
    ext = os.path.splitext(file_path)[1].lower()
    if ext in image_extensions:
        image = Image.open(file_path)
        return image
    elif ext in video_extensions:
        reader = imageio.get_reader(file_path)
        frame = reader.get_data(0)
        image = Image.fromarray(frame)
        return image
    else:
        return None

def launch_workers(argv, num_groups):
    script = os.path.abspath(argv[0])
    base_args = argv[1:]
    procs = []
    print(f"[launcher] launching {num_groups} workers")
    for gid in range(num_groups):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gid)
        child_args = [sys.executable, script] + [str(x) for x in base_args] + ['--group_id', str(gid)]
        print(f"[launcher] starting group {gid} with CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")
        p = subprocess.Popen(child_args, env=env)
        procs.append((gid, p))
    # 等待所有子进程结束
    for gid, p in procs:
        ret = p.wait()
        print(f"[launcher] worker {gid} exited with code {ret}")

def compute_group_indices(total_len, start_id, end_id, num_groups, group_id):
    if end_id == -1:
        end_id = total_len
    start_id = max(0, start_id)
    end_id = min(total_len, end_id)
    if start_id >= end_id:
        return []

    selected_len = end_id - start_id
    step = math.ceil(selected_len / num_groups)
    begin = start_id + group_id * step
    end = min(start_id + (group_id + 1) * step, end_id)
    if begin >= end:
        return []
    return list(range(begin, end))

def run_worker(cfg):
    hip_vis = os.environ.get('CUDA_VISIBLE_DEVICES', '(not set)')
    print(f"[worker {cfg.group_id}] PID={os.getpid()} CUDA_VISIBLE_DEVICES={hip_vis}")

    # Prepare output dir (per-launcher base dir plus json file name)
    out_dir = f"{cfg.out_dir}/{cfg.json_file.split('/')[-1].split('.')[0]}"
    os.makedirs(out_dir, exist_ok=True)

    with open(cfg.json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
        data_files = [f"{cfg.data_root}/{_['data_file']}" for _ in data]
        prompts = [_['prompt_en'] for _ in data]

    total_len = len(data_files)
    group_indices = compute_group_indices(total_len, cfg.start_id, cfg.end_id, cfg.num_groups, cfg.group_id)
    if len(group_indices) == 0:
        print(f"[worker {cfg.group_id}] no indices assigned, exiting.")
        return

    print(f"[worker {cfg.group_id}] assigned indices: {group_indices[0]} ... {group_indices[-1]} (count {len(group_indices)})")

    model_configs = [
        ModelConfig(model_id=cfg.model_id,
                    origin_file_pattern=f"{os.path.abspath(cfg.text_path)}",
                    local_model_path=f"{cfg.local_model_path}",
                    offload_device="cpu"),
        ModelConfig(model_id=cfg.model_id,
                    origin_file_pattern=f"{os.path.abspath(cfg.dit_path)}",
                    local_model_path=f"{cfg.local_model_path}",
                    offload_device="cpu"),
        ModelConfig(model_id=cfg.model_id,
                    origin_file_pattern=f"{os.path.abspath(cfg.vae_path)}",
                    local_model_path=f"{cfg.local_model_path}",
                    offload_device="cpu"),
    ]

    if cfg.extra_inputs == 'input_image' and cfg.img_path is not None:
        model_configs.append(
            ModelConfig(model_id=cfg.model_id,
                        origin_file_pattern=f"{os.path.abspath(cfg.img_path)}",
                        local_model_path=f"{cfg.local_model_path}",
                        offload_device="cpu"),
        )

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=model_configs,
        redirect_common_files=False,
    )

    if cfg.dit_path_full_pretrained is not None:
        state_dict = load_state_dict(cfg.dit_path_full_pretrained)
        pipe.dit.load_state_dict(state_dict)
        print(f"Loading [full] models from: {cfg.dit_path_full_pretrained}")

    if cfg.dit_path_lora is not None:
        pipe.load_lora(pipe.dit, cfg.dit_path_lora, alpha=cfg.lora_alpha)
        print(f"Loading [lora] models from: {cfg.dit_path_lora}")

    pipe.enable_vram_management()

    sizes = [[cfg.height, cfg.width, cfg.num_frames]]

    # iterate assigned indices only
    for idx in group_indices:
        data_file = data_files[idx]
        prompt = prompts[idx]
        for size in sizes:
            if cfg.extra_inputs == 'input_image' and data_file is not None:
                input_image = load_data_for_path(data_file)
                if input_image is None:
                    print(f"[worker {cfg.group_id}] Warning: cannot load {data_file} as image, skipping.")
                    continue
                input_image = input_image.resize((size[1], size[0]))
            else:
                input_image = None

            save_path = f"{out_dir}/{size[0]}_{size[1]}_{size[2]}_{idx}.mp4"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            if os.path.exists(save_path):
                continue
            print(f"[worker {cfg.group_id}] Generating idx={idx} -> {save_path} (prompt len {len(prompt)})")

            ratio = cfg.ratio
            video = pipe(
                prompt=prompt,
                negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                seed=cfg.seed,
                height=size[0], width=size[1],
                num_frames=size[2],
                input_image=input_image,
                tiled=False if cfg.tiled == 'false' else True,
                tile_size=(34 * ratio, 60 * ratio),
                tile_stride=(17 * ratio, 30 * ratio),
                cfg_scale=cfg.cfg_scale,
                cfg_merge=False if cfg.cfg_merge == 'false' else True,
                num_inference_steps=cfg.num_inference_steps,
            )
            save_video(video, save_path, fps=30, quality=8)
            print(f"[worker {cfg.group_id}] Saved {save_path}")

def main():
    cfg = parse_args()

    if cfg.group_id == -1:
        if cfg.num_groups <= 1:
            cfg.group_id = 0
            run_worker(cfg)
            return

        with open(cfg.json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        total_len = len(data)
        end_id = cfg.end_id if cfg.end_id != -1 else total_len
        start_id = max(0, cfg.start_id)
        selected_len = max(0, end_id - start_id)
        if selected_len <= 0:
            print("[launcher] no samples in the specified range, exiting.")
            return

        argv = sys.argv
        launch_workers(argv, cfg.num_groups)
        return
    else:
        run_worker(cfg)

if __name__ == '__main__':
    main()
