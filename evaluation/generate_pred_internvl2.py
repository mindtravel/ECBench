#!/usr/bin/env python3
"""
ECBench推理脚本 - InternVL2-8B
使用lmdeploy pipeline进行推理
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
from typing import Any, Dict, List
from tqdm import tqdm

from lmdeploy import pipeline, PytorchEngineConfig, ChatTemplateConfig, GenerationConfig
from lmdeploy.vl import load_image
from lmdeploy.vl.constants import IMAGE_TOKEN


SYSTEM_PROMPT = (
    "You are moving in an indoor environment. The image sequence is the scene "
    "you just saw. You are now staying at the last frame of the video. Please "
    "answer the question with one word or one sentence, as concise and accurate "
    "as possible."
)


class InternVL2Inference:
    def __init__(self, model_path: str, device: str = "cuda:0"):
        """初始化InternVL2模型"""
        print(f"Loading InternVL2 model from {model_path}...")
        
        # 针对显存优化的配置（参考EgoPlan-Bench2-code的实现）
        # 注意：新版本lmdeploy不再支持thread_safe参数
        backend_config = PytorchEngineConfig(
            tp=1,
            session_len=32768,        # 必须大幅增加，默认的通常只有 2048 或 8192
            cache_max_entry_count=0.7, # 预留 70% 显存给 KV Cache
        )
        
        # 尝试使用chat_template_config，如果失败则使用默认配置
        # InternVL2模型应该会自动检测并使用正确的模板
        try:
            chat_template_config = ChatTemplateConfig('internvl-internlm2')
            print("Using internvl-internlm2 chat template...")
            self.pipe = pipeline(
                model_path,
                chat_template_config=chat_template_config,
                backend_config=backend_config
            )
        except Exception as e:
            print(f"Warning: Could not use internvl-internlm2 template: {e}")
            print("Trying without chat_template_config (model will use default template)...")
            # 如果不支持chat_template_config，直接使用backend_config
            # InternVL2模型应该会自动检测并使用正确的模板
            self.pipe = pipeline(
                model_path,
                backend_config=backend_config
            )
        print("Model loaded successfully!")

    def inference(self, question: str, image_paths: List[str]) -> str:
        """对多张图片进行推理"""
        try:
            images = [load_image(image_path) for image_path in image_paths]
            prompt = ''
            for idx in range(len(image_paths)):
                prompt += f'Image-{str(idx + 1)}: {IMAGE_TOKEN}\n'
            prompt += question
            # 使用 min_new_tokens 避免模型立即输出 EOS 导致空结果
            gen_config = GenerationConfig(min_new_tokens=5, max_new_tokens=512)
            response = self.pipe((prompt, images), gen_config=gen_config)
            if response and hasattr(response, 'text'):
                result = response.text if response.text is not None else ""
            else:
                result = ""
            if not result:
                return ""
            return result
        except Exception as e:
            print(f"[InternVL2Inference ERROR] {repr(e)}")
            raise


def save_json(path: str, data: List[Dict[str, Any]]) -> None:
    """原子写入：先写临时文件，再重命名，避免写入中断导致文件损坏"""
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def parse_indices(s: str) -> List[int]:
    """解析序号字符串，支持: 0,3,5 或 0-9 或 0,3-5,7"""
    if not s or not s.strip():
        return []
    result = []
    for part in s.replace(" ", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            try:
                lo, hi = int(a.strip()), int(b.strip())
                result.extend(range(lo, hi + 1))
            except ValueError:
                raise ValueError(f"无效的序号范围: {part}")
        else:
            try:
                result.append(int(part))
            except ValueError:
                raise ValueError(f"无效的序号: {part}")
    return sorted(set(result))


def should_skip(item: Dict[str, Any], overwrite: bool) -> bool:
    if overwrite:
        return False
    pred = item.get("pred_answer")
    return isinstance(pred, str) and pred.strip() != ""


def find_video_path(item: Dict[str, Any], rgb_root: str) -> str:
    dataset = item.get("source_dataset", "").strip()
    video_name = item.get("video_name", "").strip()
    if not dataset or not video_name:
        return ""

    exts = [".mp4", ".avi", ".mov", ".mkv"]
    candidates = []
    if os.path.splitext(video_name)[1]:
        candidates.append(os.path.join(rgb_root, dataset, video_name))
    else:
        for ext in exts:
            candidates.append(os.path.join(rgb_root, dataset, f"{video_name}{ext}"))

    for path in candidates:
        if os.path.exists(path):
            return path
    return ""


def ffprobe_frame_count(video_path: str) -> int:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        video_path,
    ]
    try:
        out = subprocess.check_output(cmd, text=True).strip()
        return int(out)
    except Exception:
        return 0


def extract_frames(video_path: str, out_dir: str, num_frames: int) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)

    frame_count = ffprobe_frame_count(video_path)
    step = max(frame_count // num_frames, 1) if frame_count > 0 else 30

    pattern = os.path.join(out_dir, "frame_%03d.jpg")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        video_path,
        "-vf",
        f"select=not(mod(n\\,{step})),scale=768:-1",
        "-vsync",
        "vfr",
        "-q:v",
        "3",
        pattern,
    ]
    subprocess.check_call(cmd)
    frames = sorted(
        os.path.join(out_dir, x)
        for x in os.listdir(out_dir)
        if x.endswith(".jpg")
    )
    return frames[:num_frames]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate pred_answer with InternVL2-8B")
    parser.add_argument("--input", default="../data/ECbench_qa.json")
    parser.add_argument("--output", default="../data/output_internvl2_pred.json")
    parser.add_argument("--rgb-root", default="../data/rgb_video/rgb_video")
    parser.add_argument("--model-path", default="/root/autodl-tmp/InternVL2-8B")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--tmp-frames-dir", default="/tmp/ecbench_frames_internvl2")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--indices", type=str, default=None,
                        help="只处理指定序号，支持: 0,3,5 或 0-9 或 0,3-5,7")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    indices_set = None
    if args.indices is not None:
        indices_set = set(parse_indices(args.indices))
        print(f"只处理序号: {sorted(indices_set)} (共 {len(indices_set)} 个)")

    # 加载模型
    model = InternVL2Inference(args.model_path)

    # 加载数据：若 output 已存在且非 overwrite，从 output 加载以保留已有 pred_answer
    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_path != input_path and os.path.exists(output_path) and not args.overwrite:
        with open(output_path, "r", encoding="utf-8") as f:
            data: List[Dict[str, Any]] = json.load(f)
    else:
        with open(input_path, "r", encoding="utf-8") as f:
            data: List[Dict[str, Any]] = json.load(f)
        if output_path != input_path:
            save_json(output_path, data)

    if not os.path.isdir(args.rgb_root):
        raise ValueError(f"rgb video root not found: {args.rgb_root}")
    if shutil.which("ffmpeg") is None:
        raise ValueError("ffmpeg not found in PATH. Please install ffmpeg first.")

    frame_cache: Dict[str, List[str]] = {}
    processed = 0
    skipped_count = 0
    error_count = 0

    for idx, item in enumerate(tqdm(data, desc="Generating pred_answer with InternVL2")):
        if indices_set is not None and idx not in indices_set:
            continue
        if args.max_items is not None and processed >= args.max_items:
            break

        if should_skip(item, args.overwrite):
            skipped_count += 1
            continue

        question = item.get("question_en_v2.2", "").strip()
        if not question:
            item["pred_answer"] = ""
            processed += 1
            continue

        video_path = find_video_path(item, args.rgb_root)
        if not video_path:
            if args.debug:
                print(f"[WARN] idx={idx} Video not found for {item.get('video_name', 'unknown')}")
            item["pred_answer"] = ""
            processed += 1
            continue

        # 提取帧
        if video_path not in frame_cache:
            video_id = hashlib.md5(video_path.encode("utf-8")).hexdigest()[:16]
            frame_dir = os.path.join(args.tmp_frames_dir, video_id)
            if os.path.isdir(frame_dir):
                for fn in os.listdir(frame_dir):
                    if fn.endswith(".jpg"):
                        os.remove(os.path.join(frame_dir, fn))
            try:
                frame_cache[video_path] = extract_frames(
                    video_path=video_path,
                    out_dir=frame_dir,
                    num_frames=args.num_frames,
                )
            except Exception as e:
                if args.debug:
                    print(f"[ERROR] idx={idx} Failed to extract frames from {video_path}: {repr(e)}")
                item["pred_answer"] = ""
                processed += 1
                error_count += 1
                continue

        frame_paths = frame_cache.get(video_path, [])
        if not frame_paths:
            if args.debug:
                print(f"[WARN] idx={idx} No frames extracted for {video_path}")
            item["pred_answer"] = ""
            processed += 1
            continue

        # 构建prompt
        # 注意：参考EgoPlan-Bench2的实现，他们直接将question传入，不包含SYSTEM_PROMPT
        # 但ECBench要求使用SYSTEM_PROMPT，所以我们保留它
        full_question = f"{SYSTEM_PROMPT}\n\nQuestion: {question}"

        try:
            if args.debug:
                print(f"[DEBUG] idx={idx} Processing: video={video_path}, frames={len(frame_paths)}")
            answer = model.inference(full_question, frame_paths)
            item["pred_answer"] = answer
            if args.debug and answer:
                print(f"[DEBUG] idx={idx} Success: got answer length={len(answer)}")
            elif args.debug:
                print(f"[WARN] idx={idx} Got empty answer")
        except Exception as e:
            item["pred_answer"] = ""
            error_count += 1
            error_msg = f"[ERROR] idx={idx} video={video_path} video_name={item.get('video_name')} err={repr(e)}"
            print(error_msg)
            if args.debug:
                import traceback
                traceback.print_exc()
            if args.fail_fast:
                raise

        processed += 1
        if processed % args.save_every == 0:
            save_json(output_path, data)
            if args.debug:
                print(f"Saved progress: {processed}/{len(data)}, errors={error_count}, skipped={skipped_count}")
    
    print(f"\nSummary: processed={processed}, errors={error_count}, skipped={skipped_count}")

    save_json(output_path, data)
    print(f"Saved: {output_path}")
    print(f"Processed {processed} items")


if __name__ == "__main__":
    main()
