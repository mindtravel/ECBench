import argparse
import base64
import hashlib
import json
import os
import shutil
import subprocess
from typing import Any, Dict, List

from openai import OpenAI
from tqdm import tqdm


SYSTEM_PROMPT = (
    "You are moving in an indoor environment. The image sequence is the scene "
    "you just saw. You are now staying at the last frame of the video. Please "
    "answer the question with one word or one sentence, as concise and accurate "
    "as possible."
)


def build_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")

    base_url = os.environ.get("OPENAI_BASE_URL")
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def save_json(path: str, data: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


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


def image_to_data_url(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def build_user_content(question: str, frame_paths: List[str]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "The following images are sampled in temporal order from an egocentric video. "
                "Answer the question based on this sequence.\n"
                f"Question: {question}"
            ),
        }
    ]
    for path in frame_paths:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_to_data_url(path)},
            }
        )
    return content


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate pred_answer with OpenAI API")
    parser.add_argument("--input", default="../data/ECbench_qa.json")
    parser.add_argument("--output", default="../data/output_openai_pred.json")
    parser.add_argument("--rgb-root", default="../data/rgb_video")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--tmp-frames-dir", default="/tmp/ecbench_frames")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o"))
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    client = build_client()

    with open(args.input, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    if os.path.abspath(args.input) != os.path.abspath(args.output):
        save_json(args.output, data)

    if not os.path.isdir(args.rgb_root):
        raise ValueError(f"rgb video root not found: {args.rgb_root}")
    if shutil.which("ffmpeg") is None:
        raise ValueError("ffmpeg not found in PATH. Please install ffmpeg first.")

    frame_cache: Dict[str, List[str]] = {}
    processed = 0
    for item in tqdm(data, desc="Generating pred_answer"):
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
            item["pred_answer"] = ""
            processed += 1
            continue

        if video_path not in frame_cache:
            video_id = hashlib.md5(video_path.encode("utf-8")).hexdigest()[:16]
            frame_dir = os.path.join(args.tmp_frames_dir, video_id)
            if os.path.isdir(frame_dir):
                for fn in os.listdir(frame_dir):
                    if fn.endswith(".jpg"):
                        os.remove(os.path.join(frame_dir, fn))
            frame_cache[video_path] = extract_frames(
                video_path=video_path,
                out_dir=frame_dir,
                num_frames=args.num_frames,
            )

        frame_paths = frame_cache.get(video_path, [])
        if not frame_paths:
            item["pred_answer"] = ""
            processed += 1
            continue

        try:
            completion = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_content(question, frame_paths)},
                ],
                max_tokens=100,
            )
            answer = completion.choices[0].message.content or ""
            item["pred_answer"] = answer.strip()
        except Exception as e:
            item["pred_answer"] = ""
            if args.debug:
                print(f"[ERROR] idx={processed} video={video_path} model={args.model} err={repr(e)}")
            if args.fail_fast:
                raise

        processed += 1
        if processed % args.save_every == 0:
            save_json(args.output, data)

    save_json(args.output, data)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
