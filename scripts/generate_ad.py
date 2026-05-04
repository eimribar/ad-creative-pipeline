"""
Invitfull Ads Creative Factory
===============================
Generates Meta ad creatives for Invitfull by recreating ad inspirations
using Nano Banana Pro (Gemini 3 Pro Image).

Usage:
    python generate_ad.py <inspiration_image> --icp baby_shower --angle ai_wow
    python generate_ad.py inspiration.png --icp kids_birthday --angle theme_magic
    python generate_ad.py inspiration.png --system custom_instruction.md

Setup:
    1. pip install google-genai Pillow python-dotenv
    2. Set GEMINI_API_KEY in .env file
    3. Populate: logo/, website screenshots/, invitations/<icp>/
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

# --- Configuration ---

load_dotenv(Path(__file__).parent / ".env")

SCRIPT_DIR = Path(__file__).parent
LOGO_DIR = SCRIPT_DIR / "logo"
WEBSITE_DIR = SCRIPT_DIR / "website screenshots"
INVITATIONS_DIR = SCRIPT_DIR / "invitations"
OUTPUT_DIR = SCRIPT_DIR / "creatives"


def find_images(directory: Path) -> list[Path]:
    """Get all image paths sorted by filename."""
    extensions = ("*.png", "*.jpg", "*.jpeg", "*.webp")
    images = []
    for ext in extensions:
        images.extend(directory.glob(ext))
    return sorted(images)


def load_system_instruction(path: Path) -> str:
    with open(path, "r") as f:
        return f.read()


def generate_ad(
    inspiration_path: str,
    system_instruction_path: Path,
    icp: str = "",
    extra_direction: str = "",
    output_path: str | None = None,
    aspect_ratio: str = "1:1",
) -> Path | None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Set GEMINI_API_KEY in .env")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    system_instruction = load_system_instruction(system_instruction_path)

    # Find assets
    logos = find_images(LOGO_DIR)
    website_shots = find_images(WEBSITE_DIR)
    
    # ICP-specific invitations
    if icp:
        inv_dir = INVITATIONS_DIR / icp
        if not inv_dir.exists():
            # Try without subfolder (flat structure)
            inv_dir = INVITATIONS_DIR
        invitations = find_images(inv_dir)
    else:
        invitations = find_images(INVITATIONS_DIR)

    print(f"System instruction: {system_instruction_path.name}")
    print(f"Logo: {len(logos)} files")
    print(f"Website refs: {len(website_shots)} screenshots")
    print(f"Product examples: {len(invitations)} invitations")
    print(f"Inspiration: {inspiration_path}")
    if extra_direction:
        print(f"Direction: {extra_direction}")
    print("---")

    # --- Build contents array ---
    contents = []

    # 1. Logo
    if logos:
        contents.append("Invitfull brand logo:")
        contents.append(Image.open(logos[0]))

    # 2. Website screenshots (brand reference)
    for i, ss in enumerate(website_shots, 1):
        contents.append("Invitfull website screenshot for brand reference:")
        contents.append(Image.open(ss))

    # 3. Product invitation examples
    for i, inv in enumerate(invitations, 1):
        contents.append("Real Invitfull invitation example — feature this prominently:")
        contents.append(Image.open(inv))

    # 4. Ad inspiration (ALWAYS LAST image)
    contents.append("Ad inspiration to recreate for Invitfull:")
    contents.append(Image.open(inspiration_path))

    # 5. Final instruction
    instruction = "Recreate this ad for Invitfull following system instructions. Feature the invitation examples as the hero visual."
    if extra_direction:
        instruction += f" {extra_direction}"
    contents.append(instruction)

    # --- Generate ---
    print(f"Generating ad creative ({len(contents)} content parts, {sum(1 for c in contents if isinstance(c, Image.Image))} images)...")

    response = client.models.generate_content(
        model="gemini-3-pro-image-preview",
        contents=contents,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
            system_instruction=system_instruction,
            temperature=1.0,
            image_config=types.ImageConfig(aspect_ratio=aspect_ratio),
        ),
    )

    # --- Save ---
    OUTPUT_DIR.mkdir(exist_ok=True)

    if output_path is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        inspiration_name = Path(inspiration_path).stem
        icp_tag = f"_{icp}" if icp else ""
        angle_tag = f"_{system_instruction_path.stem.replace('system_instruction_', '')}"
        output_path = OUTPUT_DIR / f"invitfull{icp_tag}{angle_tag}_{inspiration_name}_{timestamp}.png"
    else:
        output_path = Path(output_path)

    for part in response.candidates[0].content.parts:
        if part.inline_data:
            with open(output_path, "wb") as f:
                f.write(part.inline_data.data)
            print(f"Ad saved to: {output_path}")
            return output_path
        elif part.text:
            print(f"Model response: {part.text}")

    print("ERROR: No image generated.")
    return None


def main():
    parser = argparse.ArgumentParser(description="Generate Invitfull ad creatives")
    parser.add_argument("inspiration", help="Path to the ad inspiration image")
    parser.add_argument("--icp", default="", help="ICP segment (baby_shower, kids_birthday, gender_reveal, general)")
    parser.add_argument("--system", "-s", default=None, help="Path to system instruction file")
    parser.add_argument("--direction", "-d", default="", help="Extra direction")
    parser.add_argument("--output", "-o", default=None, help="Custom output path")
    parser.add_argument("--ratio", "-r", default="1:1", choices=["1:1", "9:16", "16:9", "4:3", "3:4"])

    args = parser.parse_args()

    if not os.path.exists(args.inspiration):
        print(f"ERROR: Inspiration not found: {args.inspiration}")
        sys.exit(1)

    # Resolve system instruction
    if args.system:
        si_path = Path(args.system)
    else:
        si_path = SCRIPT_DIR / "system_instruction.md"

    if not si_path.exists():
        print(f"ERROR: System instruction not found: {si_path}")
        sys.exit(1)

    result = generate_ad(
        inspiration_path=args.inspiration,
        system_instruction_path=si_path,
        icp=args.icp,
        extra_direction=args.direction,
        output_path=args.output,
        aspect_ratio=args.ratio,
    )

    if result:
        print(f"\nDone! Open: {result}")
    else:
        print("\nGeneration failed.")


if __name__ == "__main__":
    main()
