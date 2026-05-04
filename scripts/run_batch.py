"""
Batch runner — generates Invitfull ads in parallel from inspirations.

Usage:
    python run_batch.py --segment baby_shower --dir "/path/to/inspirations"
    python run_batch.py --segment kids_birthday --dir "/path/to/inspirations"
    python run_batch.py --segment generic --dir "/path/to/inspirations"
    python run_batch.py --segment baby_shower --dir "/path/to/inspirations" --motivation time_pressure
"""

import asyncio
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

# --- Setup ---
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
load_dotenv(PROJECT_DIR / ".env")

SUPPORTED_RATIOS = {
    "1:1": 1.0,
    "4:3": 4 / 3,
    "3:4": 3 / 4,
    "16:9": 16 / 9,
    "9:16": 9 / 16,
}


def detect_aspect_ratio(image_path: Path) -> str:
    """Detect the closest supported Gemini aspect ratio for an image."""
    with Image.open(image_path) as img:
        w, h = img.size
    ratio = w / h
    closest = min(SUPPORTED_RATIOS, key=lambda k: abs(SUPPORTED_RATIOS[k] - ratio))
    return closest

# --- Segment configuration ---
SEGMENT_CONFIG = {
    "baby_shower": {
        "campaign_dir": PROJECT_DIR / "baby-shower",
        "logo_path": PROJECT_DIR / "baby-shower" / "reference-images" / "logo" / "logo.png",
        "screenshots_dir": PROJECT_DIR / "baby-shower" / "reference-images" / "screenshots",
        "invitations_dir": PROJECT_DIR / "baby-shower" / "reference-images" / "invitations",
        "output_dir": PROJECT_DIR / "baby-shower" / "output",
        "system_instruction_file": PROJECT_DIR / "baby-shower" / "system_instruction.md",
        "final_instruction": "Recreate this ad for Invitfull following system instructions. The product is a DIGITAL invitation platform for BABY SHOWERS. Use REAL photorealistic people — NEVER cartoon or illustrated. The words 'baby shower' MUST appear prominently in the ad.",
    },
    "kids_birthday": {
        "campaign_dir": PROJECT_DIR / "kids-birthday",
        "logo_path": PROJECT_DIR / "kids-birthday" / "reference-images" / "logo" / "Screenshot 2026-02-24 at 16.49.52.png",
        "favicon_path": PROJECT_DIR / "kids-birthday" / "reference-images" / "logo" / "favicon.png",
        "screenshots_dir": PROJECT_DIR / "kids-birthday" / "reference-images" / "screenshots",
        "invitations_dir": PROJECT_DIR / "kids-birthday" / "reference-images" / "invitations",
        "output_dir": PROJECT_DIR / "kids-birthday" / "output",
        "system_instruction_file": PROJECT_DIR / "kids-birthday" / "system_instruction.md",
        "final_instruction": "Recreate this ad for Invitfull following system instructions. The product is a DIGITAL invitation platform for KIDS BIRTHDAY PARTIES. Use REAL photorealistic people — NEVER cartoon or illustrated. The words 'birthday party' MUST appear prominently in the ad.",
    },
    "generic": {
        "campaign_dir": PROJECT_DIR / "general",
        "logo_path": PROJECT_DIR / "general" / "reference-images" / "logo" / "logo.png",
        "favicon_path": PROJECT_DIR / "general" / "reference-images" / "logo" / "logo.png",
        "screenshots_dir": PROJECT_DIR / "general" / "reference-images" / "screenshots",
        "invitations_dir": PROJECT_DIR / "general" / "reference-images" / "invitations",
        "output_dir": PROJECT_DIR / "general" / "output",
        "system_instruction_file": PROJECT_DIR / "general" / "system_instruction.md",
        "final_instruction": "Recreate this ad for Invitfull following system instructions. The product is a DIGITAL invitation platform for ANY EVENT. Use REAL photorealistic people — NEVER cartoon or illustrated.",
    },
}


def load_system_instruction(segment: str, motivation: str | None = None, instruction_override: str | None = None) -> str:
    """Load the system instruction for a segment, optionally append motivation focus."""
    if instruction_override:
        si_path = Path(instruction_override)
    else:
        si_path = SEGMENT_CONFIG[segment]["system_instruction_file"]

    if not si_path.exists():
        print(f"ERROR: System instruction not found: {si_path}")
        sys.exit(1)

    instruction = si_path.read_text()

    if motivation:
        instruction += f"\n\n## MOTIVATION FOCUS\n\nFor this generation, focus on the **{motivation.upper().replace('_', ' ')}** motivation from the framework above.\n"

    return instruction


def get_sorted_screenshots(screenshots_dir: Path) -> list[Path]:
    extensions = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.mp4")
    screenshots = []
    for ext in extensions:
        screenshots.extend(screenshots_dir.glob(ext))
    return sorted(screenshots)


def get_invitation_refs(invitations_dir: Path) -> list[Path]:
    extensions = ("*.png", "*.jpg", "*.jpeg", "*.webp")
    invitations = []
    for ext in extensions:
        invitations.extend(invitations_dir.glob(ext))
    return sorted(invitations)


def get_inspirations(inspirations_dir: Path) -> list[str]:
    """Auto-discover all image files in the inspirations directory."""
    extensions = ("*.png", "*.jpg", "*.jpeg", "*.webp")
    files = []
    for ext in extensions:
        files.extend(inspirations_dir.glob(ext))
    return sorted([f.name for f in files])


def build_contents(inspiration_path: Path, cfg: dict, index: int = 0) -> list:
    """Build the interleaved contents array for one generation."""
    import random
    contents = []

    # 1. Logo
    contents.append("This is the Invitfull brand logo. Paste this exact image into the ad. Do not redraw, re-render, or recreate this logo — use this exact image:")
    contents.append(Image.open(cfg["logo_path"]))

    # 2. Product screenshots — these ARE the product, the hero visual
    screenshots = get_sorted_screenshots(cfg["screenshots_dir"])
    for i, ss in enumerate(screenshots, 1):
        contents.append(f"Invitfull product screenshot {i}. This is what Invitfull looks like. Show these screenshots prominently in the ad as the product. Match these colors and design language:")
        contents.append(Image.open(ss))

    # 3. Example invitation designs (secondary reference)
    invitations = get_invitation_refs(cfg["invitations_dir"])
    for i, inv in enumerate(invitations, 1):
        contents.append(f"Example invitation design {i} created by Invitfull — can be featured in the ad:")
        contents.append(Image.open(inv))

    # 4. Ad inspiration (ALWAYS LAST image — model adopts its layout)
    contents.append("Ad inspiration — recreate this ad's layout and format for Invitfull. Replace the original brand and product with Invitfull's branding and product screenshots above:")
    contents.append(Image.open(inspiration_path))

    # 5. Final instruction
    contents.append(cfg["final_instruction"])

    return contents


async def generate_one(
    client,
    system_instruction: str,
    inspiration_path: Path,
    output_dir: Path,
    prefix: str,
    cfg: dict,
    index: int = 0,
    aspect_ratio: str | None = None,
) -> str:
    """Generate a single ad. If aspect_ratio is None, auto-detects from inspiration."""

    if aspect_ratio is None:
        aspect_ratio = detect_aspect_ratio(inspiration_path)

    def _sync_generate():
        contents = build_contents(inspiration_path, cfg, index=index)
        response = client.models.generate_content(
            model="gemini-3-pro-image-preview",
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
                system_instruction=system_instruction,
                temperature=1.0,
                image_config=types.ImageConfig(aspect_ratio=aspect_ratio, image_size="4K"),
            ),
        )
        return response

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, _sync_generate)

    # Save output
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    stem = inspiration_path.stem
    ratio_tag = aspect_ratio.replace(":", "x")
    output_path = output_dir / f"invitfull_{stem}_{prefix}_{ratio_tag}_{timestamp}.png"

    for part in response.candidates[0].content.parts:
        if part.inline_data:
            with open(output_path, "wb") as f:
                f.write(part.inline_data.data)
            return f"OK: {inspiration_path.name} ({aspect_ratio}) -> {output_path.name}"
        elif part.text:
            return f"TEXT RESPONSE for {inspiration_path.name} ({aspect_ratio}): {part.text[:200]}"

    return f"FAILED: {inspiration_path.name} ({aspect_ratio}) — no image generated"


async def main():
    parser = argparse.ArgumentParser(description="Batch generate Invitfull ad creatives")
    parser.add_argument(
        "--segment",
        required=True,
        choices=list(SEGMENT_CONFIG.keys()),
        help="Event segment: baby_shower, kids_birthday, generic",
    )
    parser.add_argument(
        "--dir",
        required=True,
        help="Directory containing inspiration images",
    )
    parser.add_argument(
        "--motivation",
        help="Optional motivation focus (e.g., time_pressure, creative_vision, coordination_chaos)",
    )
    parser.add_argument(
        "--output-dir",
        help="Override default output directory",
    )
    parser.add_argument(
        "--instruction",
        help="Path to a custom system instruction file (overrides segment default)",
    )
    parser.add_argument(
        "--ratios",
        default="",
        help="Comma-separated aspect ratios (e.g. '1:1,9:16'). If empty, auto-detects from inspiration.",
    )
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set")
        sys.exit(1)

    inspirations_dir = Path(args.dir)
    if not inspirations_dir.exists():
        print(f"ERROR: Inspirations directory not found: {inspirations_dir}")
        sys.exit(1)

    inspiration_names = get_inspirations(inspirations_dir)
    if not inspiration_names:
        print(f"ERROR: No image files found in {inspirations_dir}")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    cfg = dict(SEGMENT_CONFIG[args.segment])
    if args.output_dir:
        cfg["output_dir"] = Path(args.output_dir)
    system_instruction = load_system_instruction(args.segment, args.motivation, args.instruction)

    print(f"Segment: {args.segment}")
    print(f"System instruction: {cfg['system_instruction_file'].name}")
    print(f"Output: {cfg['output_dir']}")
    print(f"Logo: {cfg['logo_path']}")
    print(f"Screenshots: {len(get_sorted_screenshots(cfg['screenshots_dir']))}")
    print(f"Invitations: {len(get_invitation_refs(cfg['invitations_dir']))}")
    print(f"Inspirations: {len(inspiration_names)} from {inspirations_dir}")
    if args.motivation:
        print(f"Motivation: {args.motivation}")
    print("---")

    # Parse ratios: if empty, use None (auto-detect). Otherwise generate once per ratio.
    if args.ratios.strip():
        ratios = [r.strip() for r in args.ratios.split(",")]
        invalid = [r for r in ratios if r not in SUPPORTED_RATIOS]
        if invalid:
            print(f"ERROR: Unsupported ratios: {invalid}. Must be one of: {list(SUPPORTED_RATIOS.keys())}")
            sys.exit(1)
        print(f"Ratios: {ratios} (generating {len(ratios)} version(s) per inspiration = {len(inspiration_names) * len(ratios)} total)")
        tasks = [
            generate_one(
                client, system_instruction, inspirations_dir / name,
                cfg["output_dir"], args.segment, cfg, index=i, aspect_ratio=ratio,
            )
            for i, name in enumerate(inspiration_names)
            for ratio in ratios
        ]
    else:
        print(f"Ratios: auto-detected per inspiration")
        tasks = [
            generate_one(
                client, system_instruction, inspirations_dir / name,
                cfg["output_dir"], args.segment, cfg, index=i,
            )
            for i, name in enumerate(inspiration_names)
        ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    print(f"\nResults:")
    for r in results:
        if isinstance(r, Exception):
            print(f"  ERROR: {r}")
        else:
            print(f"  {r}")

    print(f"\nOutput folder: {cfg['output_dir']}")


if __name__ == "__main__":
    asyncio.run(main())
