"""
OpenAI variant of run_batch.py — generates Invitfull ads using OpenAI's
Responses API with the image_generation tool. Mirrors run_batch.py's CLI
and segment config so you can A/B against the Gemini pipeline.

Usage:
    python run_batch_openai.py --segment baby_shower --dir "/path/to/inspirations"
    python run_batch_openai.py --segment baby_shower --dir "/path/to/inspirations" --image-model gpt-image-2
    python run_batch_openai.py --segment baby_shower --dir "/path/to/inspirations" --reasoning-model gpt-4o

Setup:
    pip install openai Pillow python-dotenv
    OPENAI_API_KEY in <project>/.env
"""
from __future__ import annotations

import asyncio
import base64
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI
from PIL import Image

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
load_dotenv(PROJECT_DIR / ".env")

# gpt-image-* supports a small fixed set of sizes. Map aspect ratios to the
# closest supported size (square / portrait / landscape).
RATIO_TO_SIZE = {
    "1:1":  "1024x1024",
    "4:3":  "1536x1024",
    "3:4":  "1024x1536",
    "16:9": "1536x1024",
    "9:16": "1024x1536",
}
SUPPORTED_RATIOS = {
    "1:1": 1.0,
    "4:3": 4 / 3,
    "3:4": 3 / 4,
    "16:9": 16 / 9,
    "9:16": 9 / 16,
}

DEFAULT_IMAGE_MODEL = "gpt-image-2"      # falls back to gpt-image-1 if unavailable
DEFAULT_REASONING_MODEL = "gpt-4o"        # multimodal model that orchestrates the tool


def detect_aspect_ratio(image_path: Path) -> str:
    with Image.open(image_path) as img:
        w, h = img.size
    ratio = w / h
    return min(SUPPORTED_RATIOS, key=lambda k: abs(SUPPORTED_RATIOS[k] - ratio))


SEGMENT_CONFIG = {
    "baby_shower": {
        "logo_path": PROJECT_DIR / "baby-shower" / "reference-images" / "logo" / "logo.png",
        "screenshots_dir": PROJECT_DIR / "baby-shower" / "reference-images" / "screenshots",
        "invitations_dir": PROJECT_DIR / "baby-shower" / "reference-images" / "invitations",
        "output_dir": PROJECT_DIR / "baby-shower" / "output",
        "system_instruction_file": PROJECT_DIR / "baby-shower" / "system_instruction.md",
        "final_instruction": (
            "Recreate this ad for Invitfull following the system instructions. "
            "The product is a DIGITAL invitation platform for BABY SHOWERS. "
            "Use REAL photorealistic people — NEVER cartoon or illustrated. "
            "The words 'baby shower' MUST appear prominently in the ad."
        ),
    },
    "kids_birthday": {
        "logo_path": PROJECT_DIR / "kids-birthday" / "reference-images" / "logo" / "logo.png",
        "screenshots_dir": PROJECT_DIR / "kids-birthday" / "reference-images" / "screenshots",
        "invitations_dir": PROJECT_DIR / "kids-birthday" / "reference-images" / "invitations",
        "output_dir": PROJECT_DIR / "kids-birthday" / "output",
        "system_instruction_file": PROJECT_DIR / "kids-birthday" / "system_instruction.md",
        "final_instruction": (
            "Recreate this ad for Invitfull following the system instructions. "
            "The product is a DIGITAL invitation platform for KIDS BIRTHDAY PARTIES. "
            "Use REAL photorealistic people — NEVER cartoon or illustrated. "
            "The words 'birthday party' MUST appear prominently in the ad."
        ),
    },
    "generic": {
        "logo_path": PROJECT_DIR / "general" / "reference-images" / "logo" / "logo.png",
        "screenshots_dir": PROJECT_DIR / "general" / "reference-images" / "screenshots",
        "invitations_dir": PROJECT_DIR / "general" / "reference-images" / "invitations",
        "output_dir": PROJECT_DIR / "general" / "output",
        "system_instruction_file": PROJECT_DIR / "general" / "system_instruction.md",
        "final_instruction": (
            "Recreate this ad for Invitfull following the system instructions. "
            "The product is a DIGITAL invitation platform for ANY EVENT. "
            "Use REAL photorealistic people — NEVER cartoon or illustrated."
        ),
    },
}


def find_images(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    out = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        out.extend(directory.glob(ext))
    return sorted(out)


def load_system_instruction(segment: str, motivation: str | None = None,
                            override: str | None = None) -> str:
    si_path = Path(override) if override else SEGMENT_CONFIG[segment]["system_instruction_file"]
    if not si_path.exists():
        print(f"ERROR: System instruction not found: {si_path}")
        sys.exit(1)
    text = si_path.read_text()
    if motivation:
        text += (
            f"\n\n## MOTIVATION FOCUS\n\nFor this generation, focus on the "
            f"**{motivation.upper().replace('_', ' ')}** motivation.\n"
        )
    return text


def encode_image(path: Path) -> str:
    """Read image, return data URL for OpenAI input_image."""
    suffix = path.suffix.lower().lstrip(".")
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}.get(suffix, "png")
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{b64}"


def build_input(inspiration_path: Path, cfg: dict) -> list[dict]:
    """Build the multimodal user message — interleaves captions and images,
    same logical order as the Gemini script."""
    content: list[dict] = []

    # 1. Logo
    if cfg["logo_path"].exists():
        content.append({"type": "input_text", "text":
            "This is the Invitfull brand logo. Paste this exact image into the ad. "
            "Do not redraw, re-render, or recreate the logo — use this image exactly:"})
        content.append({"type": "input_image", "image_url": encode_image(cfg["logo_path"])})

    # 2. Product screenshots
    for i, ss in enumerate(find_images(cfg["screenshots_dir"]), 1):
        content.append({"type": "input_text", "text":
            f"Reference image {i} of the Invitfull product UI — for color, "
            "typography, and design-language reference only:"})
        content.append({"type": "input_image", "image_url": encode_image(ss)})

    # 3. Example invitations
    for i, inv in enumerate(find_images(cfg["invitations_dir"]), 1):
        content.append({"type": "input_text", "text":
            f"Example invitation design {i} created by Invitfull — can be featured in the ad:"})
        content.append({"type": "input_image", "image_url": encode_image(inv)})

    # 4. Inspiration (always last image)
    content.append({"type": "input_text", "text":
        "Ad inspiration — recreate this ad's layout and format for Invitfull. "
        "Replace the original brand and product with Invitfull's branding and "
        "product screenshots above:"})
    content.append({"type": "input_image", "image_url": encode_image(inspiration_path)})

    # 5. Final instruction
    content.append({"type": "input_text", "text": cfg["final_instruction"]})

    return content


def resolve_image_model(client: OpenAI, requested: str) -> str:
    """Probe the models endpoint. If the requested image model isn't listed,
    fall back to gpt-image-1."""
    try:
        models = {m.id for m in client.models.list().data}
    except Exception as e:
        print(f"WARN: could not list models ({e}); using {requested} as-is")
        return requested
    if requested in models:
        return requested
    if "gpt-image-1" in models:
        print(f"WARN: '{requested}' not available on this key; falling back to gpt-image-1")
        return "gpt-image-1"
    print(f"ERROR: no gpt-image-* model found on this key. Available: "
          f"{sorted(m for m in models if 'image' in m)}")
    sys.exit(1)


async def generate_one(
    client: AsyncOpenAI,
    system_instruction: str,
    inspiration_path: Path,
    output_dir: Path,
    prefix: str,
    cfg: dict,
    image_model: str,
    reasoning_model: str,
    aspect_ratio: str | None = None,
) -> str:
    if aspect_ratio is None:
        aspect_ratio = detect_aspect_ratio(inspiration_path)
    size = RATIO_TO_SIZE[aspect_ratio]

    user_content = build_input(inspiration_path, cfg)

    try:
        response = await client.responses.create(
            model=reasoning_model,
            input=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_content},
            ],
            tools=[{
                "type": "image_generation",
                "model": image_model,
                "size": size,
                "quality": "high",
            }],
            tool_choice={"type": "image_generation"},
        )
    except Exception as e:
        return f"FAILED: {inspiration_path.name} ({aspect_ratio}) — {type(e).__name__}: {e}"

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    stem = inspiration_path.stem
    ratio_tag = aspect_ratio.replace(":", "x")
    output_path = output_dir / f"invitfull_{stem}_{prefix}_{ratio_tag}_{timestamp}.png"

    image_b64 = None
    text_msg = None
    for item in response.output:
        kind = getattr(item, "type", None)
        if kind == "image_generation_call":
            image_b64 = getattr(item, "result", None)
            if image_b64:
                break
        elif kind == "message":
            for part in getattr(item, "content", []) or []:
                if getattr(part, "type", None) == "output_text":
                    text_msg = getattr(part, "text", None)

    if image_b64:
        output_path.write_bytes(base64.b64decode(image_b64))
        return f"OK: {inspiration_path.name} ({aspect_ratio}) -> {output_path.name}"
    if text_msg:
        return f"TEXT RESPONSE for {inspiration_path.name}: {text_msg[:200]}"
    return f"FAILED: {inspiration_path.name} ({aspect_ratio}) — no image in response"


async def run(args):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set in .env")
        sys.exit(1)

    inspirations_dir = Path(args.dir)
    if not inspirations_dir.exists():
        print(f"ERROR: Inspirations directory not found: {inspirations_dir}")
        sys.exit(1)

    inspiration_paths = find_images(inspirations_dir)
    if not inspiration_paths:
        print(f"ERROR: No image files in {inspirations_dir}")
        sys.exit(1)

    cfg = dict(SEGMENT_CONFIG[args.segment])
    if args.output_dir:
        cfg["output_dir"] = Path(args.output_dir)

    system_instruction = load_system_instruction(args.segment, args.motivation, args.instruction)

    sync_client = OpenAI(api_key=api_key)
    image_model = resolve_image_model(sync_client, args.image_model)

    print(f"Segment: {args.segment}")
    print(f"System instruction: {cfg['system_instruction_file'].name}")
    print(f"Reasoning model: {args.reasoning_model}")
    print(f"Image model: {image_model}")
    print(f"Output: {cfg['output_dir']}")
    print(f"Logo: {cfg['logo_path']} (exists: {cfg['logo_path'].exists()})")
    print(f"Screenshots: {len(find_images(cfg['screenshots_dir']))}")
    print(f"Invitations: {len(find_images(cfg['invitations_dir']))}")
    print(f"Inspirations: {len(inspiration_paths)} from {inspirations_dir}")
    if args.motivation:
        print(f"Motivation: {args.motivation}")
    print("---")

    async_client = AsyncOpenAI(api_key=api_key)

    if args.ratios.strip():
        ratios = [r.strip() for r in args.ratios.split(",")]
        bad = [r for r in ratios if r not in SUPPORTED_RATIOS]
        if bad:
            print(f"ERROR: Unsupported ratios: {bad}. Use: {list(SUPPORTED_RATIOS)}")
            sys.exit(1)
        print(f"Ratios: {ratios} ({len(inspiration_paths) * len(ratios)} generations total)")
        tasks = [
            generate_one(async_client, system_instruction, p, cfg["output_dir"],
                         args.segment, cfg, image_model, args.reasoning_model, ratio)
            for p in inspiration_paths
            for ratio in ratios
        ]
    else:
        print("Ratios: auto-detected per inspiration")
        tasks = [
            generate_one(async_client, system_instruction, p, cfg["output_dir"],
                         args.segment, cfg, image_model, args.reasoning_model)
            for p in inspiration_paths
        ]

    sem = asyncio.Semaphore(args.concurrency)
    async def gated(task):
        async with sem:
            return await task

    results = await asyncio.gather(*(gated(t) for t in tasks), return_exceptions=True)

    print("\nResults:")
    for r in results:
        print(f"  ERROR: {r}" if isinstance(r, Exception) else f"  {r}")
    print(f"\nOutput folder: {cfg['output_dir']}")


def main():
    parser = argparse.ArgumentParser(description="Batch generate Invitfull ads via OpenAI")
    parser.add_argument("--segment", required=True, choices=list(SEGMENT_CONFIG))
    parser.add_argument("--dir", required=True, help="Directory of inspiration images")
    parser.add_argument("--motivation", help="Optional motivation focus appended to the system instruction")
    parser.add_argument("--output-dir", help="Override output directory")
    parser.add_argument("--instruction", help="Path to a custom system instruction file")
    parser.add_argument("--ratios", default="", help="Comma-separated ratios (e.g. '1:1,9:16'). Empty = auto-detect.")
    parser.add_argument("--image-model", default=DEFAULT_IMAGE_MODEL, help="Image generation model")
    parser.add_argument("--reasoning-model", default=DEFAULT_REASONING_MODEL, help="Multimodal model that drives the image_generation tool")
    parser.add_argument("--concurrency", type=int, default=4, help="Max parallel generations")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
