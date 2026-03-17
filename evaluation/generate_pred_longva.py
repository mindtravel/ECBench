#!/usr/bin/env python3
"""
ECBench推理脚本 - LongVA-7B-DPO
使用LongVA模型进行推理
"""
import argparse
import json
import os
import sys
from typing import Any, Dict, List
from tqdm import tqdm

# 添加LongVA路径到sys.path
sys.path.insert(0, '/root/autodl-tmp/EgoPlan-Bench2-code/models/LongVA')

from inference import LongVA, video_demo


SYSTEM_PROMPT = (
    "You are moving in an indoor environment. The image sequence is the scene "
    "you just saw. You are now staying at the last frame of the video. Please "
    "answer the question with one word or one sentence, as concise and accurate "
    "as possible."
)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate pred_answer with LongVA-7B-DPO")
    parser.add_argument("--input", default="../data/ECbench_qa.json")
    parser.add_argument("--output", default="../data/output_longva_pred.json")
    parser.add_argument("--rgb-root", default="../data/rgb_video/rgb_video")
    parser.add_argument("--model-path", default="/root/autodl-tmp/LongVA-7B-DPO")
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--indices", type=str, default=None,
                        help="只处理指定序号，支持: 0,3,5 或 0-9 或 0,3-5,7")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    indices_set = None
    if args.indices is not None:
        indices_set = set(parse_indices(args.indices))
        print(f"只处理序号: {sorted(indices_set)} (共 {len(indices_set)} 个)")

    # 加载模型
    print(f"Loading LongVA model from {args.model_path}...")
    model = LongVA(
        pretrained=args.model_path,
        model_name="llava_qwen",
        device_map=args.device,
        max_frames_num=args.num_frames
    )
    print("Model loaded successfully!")

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

    processed = 0
    for idx, item in enumerate(tqdm(data, desc="Generating pred_answer with LongVA")):
        if indices_set is not None and idx not in indices_set:
            continue
        if args.max_items is not None and processed >= args.max_items:
            break
        if should_skip(item, args.overwrite):
            continue

        question = item.get("question_en_v2.2", "").strip()
        if not question:
            item["pred_answer"] = ""
            processed += 1
            continue

        video_path = find_video_path(item, args.rgb_root)
        if not video_path:
            if args.debug:
                print(f"[WARN] Video not found for {item.get('video_name', 'unknown')}")
            item["pred_answer"] = ""
            processed += 1
            continue

        # 构建完整问题
        full_question = f"{SYSTEM_PROMPT}\n\nQuestion: {question}"

        try:
            # 使用LongVA进行推理
            # 构建请求
            input_visuals = [video_path]
            input_context = full_question
            task_type = "video"
            gen_kwargs = {
                "max_new_tokens": 1024,
                "temperature": 0,
                "do_sample": False,
                "sample_frames": args.num_frames
            }
            query = {
                "visuals": input_visuals,
                "context": input_context,
                "task_type": task_type,
                "prev_conv": [],
            }
            
            # 生成答案（stream_generate_until 是 LongVA 类的方法，不是 model 的）
            generated_text = ""
            for x in model.stream_generate_until(query, gen_kwargs):
                output = json.loads(x.decode("utf-8").strip("\0"))["text"].strip()
                generated_text = output
            
            item["pred_answer"] = generated_text.strip()
        except Exception as e:
            item["pred_answer"] = ""
            if args.debug:
                print(f"[ERROR] idx={processed} video={video_path} err={repr(e)}")
                import traceback
                traceback.print_exc()
            if args.fail_fast:
                raise

        processed += 1
        if processed % args.save_every == 0:
            save_json(output_path, data)
            if args.debug:
                print(f"Saved progress: {processed}/{len(data)}")

    save_json(output_path, data)
    print(f"Saved: {output_path}")
    print(f"Processed {processed} items")


if __name__ == "__main__":
    main()
