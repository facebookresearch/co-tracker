"""VLM quality check of one video: yes/no questions about its frames, asked one at a time to a local
Qwen2.5-VL or Qwen3-VL model.

Each question gets its own images (see "Questions") and is scored by the model's probability of
answering "Yes" (P(yes), 0..1) instead of a generated Y/N: asked for a plain Y/N the model says N to
nearly everything, but P(yes) still ranks bad videos above good ones. A question whose P(yes) is above
its threshold is flagged ("Y"); the anomaly score is the fraction of flagged questions (0 = no issue
found, 1 = every issue found). The batch pipeline runs it on video B of every demo
(scripts/batch_pipeline.py, step "vlm").

Questions: a JSON list (default scripts/vlm_questions.json) of
    {"id": "towel_patch",            letters, digits and underscores; the column name in the GUI table
     "question": "...",              asked as "<question> Answer Yes or No."
     "view": "third_person_view_object",  what the model sees, default "both_views":
                                       both_views   the whole frame (third-person and wrist view side by side)
                                       third_person_view, wrist_view
                                                    the left (third-person) / right (wrist) half of the frame
                                       third_person_view_object, wrist_view_object
                                                    a square close-up of the object in that half, found
                                                    with the instance video (towel_color); the whole
                                                    half when there is no instance video or no object
                                     The old names both, room, wrist, room_towel and wrist_towel are still
                                     accepted and mean the same views.
     "at": [0.55, 0.7, 0.85, 1.0],   which frames, as fractions of the video (0 = first, 1 = last),
                                     default [0, 0.5, 1]
     "threshold": 0.05}              flag when P(yes) > threshold, default 0.5
The view names assume the Cosmos split-screen layout: third-person view left, wrist view right.
The thresholds of the default questions were tuned with Qwen2.5-VL-7B on 240 reviewed O/X videos
(franka_towel_n750_seeds1234); another model needs other thresholds.

Model weights: checkpoints/vlm/ (Hugging Face cache layout); a model that is not there yet is
downloaded into it.

Single video:
    python scripts/vlm_check.py --video path/to/demo_001/demo_001_01.mp4 [--device cuda:0]
        [--instance path/to/demo_001_instance_all.mp4] [--questions q.json]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_QUESTIONS = REPO / "scripts" / "vlm_questions.json"
MODEL_CACHE = REPO / "checkpoints" / "vlm"
QUESTION_ID = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
VIEWS = ("both_views", "third_person_view", "wrist_view", "third_person_view_object", "wrist_view_object")
VIEW_ALIASES = {"both": "both_views", "room": "third_person_view", "wrist": "wrist_view",
                "room_towel": "third_person_view_object", "wrist_towel": "wrist_view_object"}  # old names
TOWEL_COLOR = (167, 18, 135)  # the towel in the SoftMimicGen instance videos
CROP_SIZE = 448  # side of a towel close-up, px


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    view: str = "both_views"
    at: tuple[float, ...] = (0.0, 0.5, 1.0)
    threshold: float = 0.5

    def to_json(self):
        return {"id": self.id, "question": self.text, "view": self.view, "at": list(self.at),
                "threshold": self.threshold}


def load_questions(path: Path) -> list[Question]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read questions from {path}: {exc}") from exc
    if not isinstance(raw, list) or not raw:
        raise ValueError("Questions file must contain a non-empty JSON list")
    questions: list[Question] = []
    seen: set[str] = set()
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Question {index} must be a JSON object")
        question_id, text = item.get("id"), item.get("question")
        if not isinstance(question_id, str) or not QUESTION_ID.fullmatch(question_id):
            raise ValueError(f"Question {index} has invalid id {question_id!r}; use letters, digits, and underscores")
        if question_id in seen:
            raise ValueError(f"Duplicate question id: {question_id}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Question {index} has empty question text")
        view = item.get("view", "both_views")
        view = VIEW_ALIASES.get(view, view)
        if view not in VIEWS:
            raise ValueError(f"Question {question_id}: view must be one of {', '.join(VIEWS)}")
        at = item.get("at", [0.0, 0.5, 1.0])
        if (not isinstance(at, list) or not at
                or not all(isinstance(a, (int, float)) and 0 <= a <= 1 for a in at)):
            raise ValueError(f"Question {question_id}: at must be a non-empty list of fractions 0..1")
        threshold = item.get("threshold", 0.5)
        if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
            raise ValueError(f"Question {question_id}: threshold must be a number 0..1")
        questions.append(Question(question_id, text.strip(), view, tuple(float(a) for a in at), float(threshold)))
        seen.add(question_id)
    return questions


def run_checked(command: list[str]) -> str:
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required executable not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Command failed ({command[0]}): {(exc.stderr or exc.stdout).strip()}") from exc
    return result.stdout


def get_frame_count(video: Path) -> int:
    # The stream metadata usually has the exact count; decode the stream only when it is missing.
    probe = run_checked(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                         "stream=nb_frames", "-of", "json", str(video)])
    try:
        count = int(json.loads(probe)["streams"][0]["nb_frames"])
        if count > 0:
            return count
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
        pass
    probe = run_checked(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                         "-show_entries", "stream=nb_read_frames", "-of", "json", str(video)])
    try:
        count = int(json.loads(probe)["streams"][0]["nb_read_frames"])
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not determine frame count for {video}") from exc
    if count < 1:
        raise RuntimeError(f"Video has no readable frames: {video}")
    return count


def read_frames(video: Path, indexes: set[int]) -> dict[int, np.ndarray]:
    """The frames at `indexes` as RGB arrays, in one decoding pass."""
    import av

    frames: dict[int, np.ndarray] = {}
    last = max(indexes)
    with av.open(str(video)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i in indexes:
                frames[i] = frame.to_ndarray(format="rgb24")
            if i >= last:
                break
    missing = indexes - set(frames)
    if missing:
        raise RuntimeError(f"{video.name} has no frame {min(missing)}")
    return frames


def frame_indexes(question: Question, count: int) -> list[int]:
    return sorted({round(a * (count - 1)) for a in question.at})


def towel_box(instance: np.ndarray, color, tolerance=40, pad=0.25):
    """Square (x0, y0, x1, y1) around the towel pixels of an instance image, or None without a towel.
    color: one (R, G, B) or a list of them; a pixel close to any of them is a towel pixel (the instance
    color of the towel can differ from video to video)."""
    colors = np.asarray(color).reshape(-1, 3)
    mask = (np.abs(instance.astype(int)[..., None, :] - colors).sum(-1) < tolerance).any(-1)
    if mask.sum() < 50:
        return None
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    side = min(max(xs.max() - xs.min(), ys.max() - ys.min(), 96) * (1 + 2 * pad), h, w)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    x0 = int(min(max(cx - side / 2, 0), w - side))
    y0 = int(min(max(cy - side / 2, 0), h - side))
    return x0, y0, x0 + int(side), y0 + int(side)


def question_images(video: Path, questions: list[Question], out_dir: Path, instance: Path | None = None,
                    towel_color=TOWEL_COLOR) -> dict[str, list[Path]]:
    """{question id: [JPEG paths]}: the images each question is asked with, saved under
    out_dir/<video stem>/<view>/<frame>.jpg. The video (and instance video) is decoded once."""
    from PIL import Image

    count = get_frame_count(video)
    wanted = {i for q in questions for i in frame_indexes(q, count)}
    frames = read_frames(video, wanted)
    towel_views = any(q.view.endswith("_object") for q in questions)
    inst = {}
    if towel_views and instance is not None and instance.is_file():
        inst_count = get_frame_count(instance)
        inst = read_frames(instance, {min(i, inst_count - 1) for i in wanted})
        inst = {i: inst[min(i, inst_count - 1)] for i in wanted}
    images: dict[str, list[Path]] = {}
    for q in questions:
        d = out_dir / video.stem / q.view
        d.mkdir(parents=True, exist_ok=True)
        paths = []
        for i in frame_indexes(q, count):
            path = d / f"{i:04d}.jpg"
            if not path.is_file():
                img = frames[i]
                if q.view != "both_views":
                    half = img.shape[1] // 2
                    x = 0 if q.view.startswith("third_person") else half
                    img = img[:, x:x + half]
                    if q.view.endswith("_object"):
                        box = towel_box(inst[i][:, x:x + half], towel_color) if i in inst else None
                        if box is not None:
                            img = np.asarray(Image.fromarray(img).crop(box).resize((CROP_SIZE, CROP_SIZE),
                                                                                  Image.BICUBIC))
                Image.fromarray(np.ascontiguousarray(img)).save(path, quality=95)
            paths.append(path)
        images[q.id] = paths
    return images


def find_instance(video: Path) -> Path | None:
    """demo_XXX_instance_all.mp4 or demo_XXX_instance.mp4 next to demo_XXX_*.mp4."""
    demo = video.parent.name if video.parent.name.startswith("demo_") else "_".join(video.stem.split("_")[:2])
    for suffix in ("_instance_all", "_instance"):
        p = video.with_name(f"{demo}{suffix}.mp4")
        if p.is_file():
            return p
    return None


def anomaly_score(answers: dict[str, dict]) -> float:
    return sum(item["answer"] == "Y" for item in answers.values()) / len(answers)


class QwenChecker:
    """Qwen2.5-VL or Qwen3-VL (a hub name or a local folder) on one device ("cpu", "cuda:N"), loaded once and
    reused for every video."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu", model_cache: Path = MODEL_CACHE):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch, self.device = torch, device
        model_cache.mkdir(parents=True, exist_ok=True)

        def load(local_only):
            processor = AutoProcessor.from_pretrained(model_name, cache_dir=model_cache, local_files_only=local_only)
            model = AutoModelForImageTextToText.from_pretrained(
                model_name, torch_dtype="auto", attn_implementation="sdpa", cache_dir=model_cache,
                local_files_only=local_only)
            return processor, model

        try:
            self.processor, model = load(True)
        except OSError:  # not in checkpoints/vlm yet: download it there
            print(f"Downloading {model_name} into {model_cache}...", flush=True)
            self.processor, model = load(False)
        self.model = model.to(device).eval()
        tok = self.processor.tokenizer
        self.yes_ids = [tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "yes")]
        self.no_ids = [tok.encode(w, add_special_tokens=False)[0] for w in ("No", "no")]

    def p_yes(self, images: list[Path], question: str) -> float:
        """P("Yes") / (P("Yes") + P("No")) for the first answer token."""
        from qwen_vl_utils import process_vision_info

        content = [{"type": "image", "image": p.resolve().as_uri()} for p in images]
        content.append({"type": "text", "text": f"{question} Answer Yes or No."})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # the images are resized in steps of the model's patch size (14 for Qwen2.5-VL, 16 for Qwen3-VL)
        image_inputs, video_inputs = process_vision_info(
            messages, image_patch_size=self.processor.image_processor.patch_size)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs,
                                return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits[0, -1].float()
        logp = self.torch.log_softmax(logits, -1)
        yes, no = self.torch.logsumexp(logp[self.yes_ids], 0), self.torch.logsumexp(logp[self.no_ids], 0)
        return self.torch.sigmoid(yes - no).item()

    def ask(self, images: dict[str, list[Path]], questions: list[Question]) -> dict[str, dict]:
        """{id: {"p_yes", "threshold", "answer": "Y" if p_yes > threshold else "N"}}."""
        answers = {}
        for q in questions:
            p = self.p_yes(images[q.id], q.text)
            answers[q.id] = {"p_yes": p, "threshold": q.threshold, "answer": "Y" if p > q.threshold else "N"}
        return answers


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--instance", type=Path, help="instance video for the *_object views "
                   "(default: demo_XXX_instance_all.mp4 or demo_XXX_instance.mp4 next to the video)")
    p.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--device", default="cuda:0", help="cpu or cuda:N")
    p.add_argument("--frames_dir", type=Path, default=REPO / "outputs" / "vlm_frames")
    args = p.parse_args()
    questions = load_questions(args.questions)
    images = question_images(args.video, questions, args.frames_dir, args.instance or find_instance(args.video))
    answers = QwenChecker(args.model, args.device).ask(images, questions)
    for q in questions:
        a = answers[q.id]
        print(f"{q.id}: {a['answer']}  P(yes) {a['p_yes']:.3f} (threshold {q.threshold:g})")
    print(f"anomaly score: {anomaly_score(answers):.3f}")


if __name__ == "__main__":
    main()
