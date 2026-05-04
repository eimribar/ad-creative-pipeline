"""
Two-stage carousel generation for Meta (IG/FB) — panel-grid mode.

Stage 1 (content): a content-writing model (default gpt-5.5) reads
system_instruction_carousel.md + the topic and returns a JSON-schema-validated
content brief. Content only — no design specification.

Stage 2 (render): ONE Responses-API call with the image_generation tool produces
the full carousel as a single composite image: an N-panel grid laid out cols×rows.
The model commits to one design system across all panels natively. The composite
is then split into N individual 1080×1080 PNGs.

Usage:
    python scripts/run_carousel_openai.py \\
        --segment baby_shower \\
        --topic "9 baby shower trends taking over 2026" \\
        --slides 9
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from PIL import Image

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
load_dotenv(PROJECT_DIR / ".env")

DEFAULT_CONTENT_MODEL = "gpt-5.5"
DEFAULT_REASONING_MODEL = "gpt-4o"
DEFAULT_IMAGE_MODEL = "gpt-image-2"
META_SLIDE_PX = 1080
# Per-panel size chosen so total grid (panel × cols, panel × rows):
#   - stays under gpt-image-2's documented reliability ceiling of ~3.7M pixels
#   - has both edges divisible by 16 (hard API requirement).
# 608 is divisible by 16 (608 = 16 × 38). 9 panels at 608² = 1824² = 3.33M ✓.
PANEL_NATIVE_PX = 608
DEFAULT_STYLE_REFS_DIR = Path("/Users/eimri/Desktop/CAROUSEL EXAMPLE")
DEFAULT_DESIGN_SPEC = Path("/Users/eimri/Desktop/invitfull-design.md")

# Grid layouts chosen so total dimensions stay under gpt-image-2's 3840 max-edge
# and 3:1 ratio cap. (cols, rows). Picked to minimize empty cells.
GRID_LAYOUTS: dict[int, tuple[int, int]] = {
    1:  (1, 1),
    2:  (2, 1),
    3:  (3, 1),
    4:  (2, 2),
    5:  (3, 2),
    6:  (3, 2),
    7:  (4, 2),
    8:  (4, 2),
    9:  (3, 3),
    10: (5, 2),
}

SEGMENT_CONFIG = {
    "baby_shower": {
        "logo_path": PROJECT_DIR / "baby-shower" / "reference-images" / "logo" / "logo.png",
        "screenshots_dir": PROJECT_DIR / "baby-shower" / "reference-images" / "screenshots",
        "invitations_dir": PROJECT_DIR / "baby-shower" / "reference-images" / "invitations",
        "output_root": PROJECT_DIR / "baby-shower" / "output",
        "carousel_si_file": PROJECT_DIR / "baby-shower" / "system_instruction_carousel.md",
    },
    "kids_birthday": {
        "logo_path": PROJECT_DIR / "kids-birthday" / "reference-images" / "logo" / "logo.png",
        "screenshots_dir": PROJECT_DIR / "kids-birthday" / "reference-images" / "screenshots",
        "invitations_dir": PROJECT_DIR / "kids-birthday" / "reference-images" / "invitations",
        "output_root": PROJECT_DIR / "kids-birthday" / "output",
        "carousel_si_file": PROJECT_DIR / "kids-birthday" / "system_instruction_carousel.md",
    },
    "generic": {
        "logo_path": PROJECT_DIR / "general" / "reference-images" / "logo" / "logo.png",
        "screenshots_dir": PROJECT_DIR / "general" / "reference-images" / "screenshots",
        "invitations_dir": PROJECT_DIR / "general" / "reference-images" / "invitations",
        "output_root": PROJECT_DIR / "general" / "output",
        "carousel_si_file": PROJECT_DIR / "general" / "system_instruction_carousel.md",
    },
}

CAROUSEL_BRIEF_SCHEMA = {
    "type": "object",
    "properties": {
        "topic": {
            "type": "string",
            "description": "The listicle premise — same as or sharpened from the user's input.",
        },
        "slides": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "n": {"type": "integer"},
                    "role": {"type": "string", "enum": ["hook", "concept", "outro"]},
                    "label": {
                        "type": "string",
                        "description": "The big headline text rendered on the slide. 2-5 words for concept slides; can be longer for hook.",
                    },
                    "body": {
                        "type": "string",
                        "description": "Optional 1-2 sentence editorial subhead. Empty string for hook. Required for concept and outro.",
                    },
                    "photo_subject": {
                        "type": "string",
                        "description": "1-3 word description of what the photo on this slide depicts. Content only, never design.",
                    },
                },
                "required": ["n", "role", "label", "body", "photo_subject"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["topic", "slides"],
    "additionalProperties": False,
}


# ───────────────────────────── helpers ─────────────────────────────

def find_images(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    out: list[Path] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        out.extend(directory.glob(ext))
    return sorted(out)


def encode_image(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}.get(suffix, "png")
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{b64}"


def encode_bytes(data: bytes, mime: str = "png") -> str:
    return f"data:image/{mime};base64,{base64.b64encode(data).decode('ascii')}"


def brand_dna_blocks(cfg: dict, n_slides: int, enabled: bool = True,
                     design_spec_path: Path | None = None) -> list[dict]:
    """Brand assets + (optional) design spec markdown passed for color +
    typographic DNA. Image references never to be pasted into slides.
    Logo only appears on the outro panel."""
    if not enabled:
        return []
    blocks: list[dict] = []
    if design_spec_path and design_spec_path.exists():
        blocks.append({"type": "input_text", "text":
            "BRAND DESIGN SYSTEM SPEC (Invitfull / Poppy). Treat sections 1-3 "
            "and 11-12 as hard rules. Honor the color palette (hex codes), "
            "typography (PP Frama Bold for ALL headlines, General Sans for "
            "body), composition recipes (named patterns), eyebrow pill "
            "convention, three-layer rule, geometry containment, and the "
            "'Handled' voice. Apply this spec to every panel:\n\n"
            + design_spec_path.read_text()})
    pieces: list[Path] = []
    if cfg["logo_path"].exists():
        pieces.append(cfg["logo_path"])
    pieces.extend(find_images(cfg["screenshots_dir"]))
    if not pieces:
        return blocks
    blocks.append({"type": "input_text", "text":
        f"Brand DNA references for Invitfull. Pull the carousel's accent color "
        f"and typographic energy from these — same color family, same type spirit. "
        f"DO NOT paste any of these images into any panel. "
        f"The Invitfull logo may appear ONLY on panel {n_slides} (the outro), "
        f"small and tasteful. Panels 1..{n_slides - 1} contain NO logo."})
    for p in pieces:
        blocks.append({"type": "input_image", "image_url": encode_image(p)})
    return blocks


def style_reference_blocks(refs_dir: Path | None) -> list[dict]:
    """Pass example viral carousels as visual energy references.
    Tells the model: match this quality / vibe, but make original design choices."""
    if refs_dir is None or not refs_dir.exists():
        return []
    examples = []
    # Each subfolder is one example carousel. Pull a couple of slides from each.
    for sub in sorted(p for p in refs_dir.iterdir() if p.is_dir()):
        slides = find_images(sub)
        if not slides:
            continue
        # Cap at 3 per example to keep the prompt reasonable
        for s in slides[:3]:
            examples.append(s)
    if not examples:
        return []
    blocks: list[dict] = [{"type": "input_text", "text":
        "Reference carousels showing the kind of viral organic Instagram editorial "
        "carousel we're aiming for. Match this LEVEL of design quality, energy, "
        "and 'feels-like-a-magazine-not-an-ad' tone. DO NOT copy these designs — "
        "make ORIGINAL creative choices. These are quality bar / vibe references only:"}]
    for img in examples:
        blocks.append({"type": "input_image", "image_url": encode_image(img)})
    return blocks


# ───────────────────────────── Stage 1 ─────────────────────────────

CAROUSEL_BRIEF_TASK = """\
Write a {n}-slide viral organic Instagram carousel for Invitfull on this topic:

  TOPIC: {topic}

This is editorial baby-shower content — a magazine-style listicle. Useful, specific, share-worthy. NOT an ad.

Slide structure:
- Slide 1 = HOOK. role="hook". label = the full topic phrased as a bold declaration. body = "". photo_subject = something that visually anchors the topic.
- Slides 2..{nm1} = CONCEPT. role="concept". label = 2-5 words, the concept name. body = 1-2 sentences explaining it. photo_subject = 1-3 words for the photo content.
- Slide {n} = OUTRO. role="outro". label = a confident soft Invitfull line ("Plan yours with Invitfull" or similar). body = one sentence. photo_subject = something celebratory.

Hard rules:
- The middle (concept) slides are CONTENT, not pitch. DO NOT mention Invitfull in slides 2..{nm1}.
- The outro is the ONLY slide that mentions Invitfull.
- "Baby shower" appears in slide 1 (hook) and slide {n} (outro) at minimum.
- No marketing speak. Specific over general.
- DO NOT specify visual design — only content (label, body, photo_subject).
"""


async def stage1_brief(
    client: AsyncOpenAI,
    *,
    content_model: str,
    system_instruction: str,
    topic: str,
    n_slides: int,
) -> dict:
    user_content = [{"type": "input_text", "text":
        CAROUSEL_BRIEF_TASK.format(topic=topic, n=n_slides, nm1=n_slides - 1)}]

    response = await client.responses.create(
        model=content_model,
        input=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "carousel_brief",
                "schema": CAROUSEL_BRIEF_SCHEMA,
                "strict": True,
            },
        },
    )

    text = ""
    for item in response.output:
        if getattr(item, "type", None) == "message":
            for part in getattr(item, "content", []) or []:
                if getattr(part, "type", None) == "output_text":
                    text += getattr(part, "text", "") or ""
    if not text:
        text = getattr(response, "output_text", "") or ""
    if not text:
        raise RuntimeError("Stage 1 returned no text output")

    brief = json.loads(text)
    if len(brief["slides"]) != n_slides:
        raise RuntimeError(
            f"Stage 1 returned {len(brief['slides'])} slides, expected {n_slides}"
        )
    return brief


# ───────────────────────────── Stage 2 ─────────────────────────────

def panel_layout_description(n: int) -> tuple[int, int, int, str]:
    """Return (cols, rows, empty_cells, layout_description_text)."""
    if n not in GRID_LAYOUTS:
        raise ValueError(f"Unsupported slide count: {n}. Pick 1..10.")
    cols, rows = GRID_LAYOUTS[n]
    total_cells = cols * rows
    empty = total_cells - n
    desc = (
        f"{cols}-column by {rows}-row grid, "
        f"{n} content panels numbered left-to-right, top-to-bottom"
    )
    if empty > 0:
        desc += (f", with {empty} cell{'s' if empty != 1 else ''} unused "
                 f"(render as solid background, no content)")
    return cols, rows, empty, desc


def build_render_prompt(brief: dict, n: int) -> str:
    cols, rows, empty, layout_desc = panel_layout_description(n)
    panel_size_px = PANEL_NATIVE_PX
    grid_w = cols * panel_size_px
    grid_h = rows * panel_size_px

    panel_lines = []
    for s in brief["slides"]:
        eyebrow_line = (f"  Eyebrow tag (small label rendered above the headline): \"{s['eyebrow']}\"\n"
                        if s.get("eyebrow") else "")
        panel_lines.append(
            f"PANEL {s['n']} ({s['role']}):\n"
            + eyebrow_line
            + f"  Headline (render this exact text, prominently): \"{s['label']}\"\n"
            + (f"  Body (render this exact text, smaller, beneath the headline): \"{s['body']}\"\n"
               if s['body'] else "  Body: (none)\n")
            + f"  Photo: {s['photo_subject']}"
        )
    panels_block = "\n\n".join(panel_lines)

    return f"""Render this {n}-slide viral organic Instagram carousel as a SINGLE composite image — a {layout_desc}, total dimensions {grid_w}×{grid_h} pixels.

This is editorial baby-shower content — a magazine-style listicle, NOT an ad.

Topic: {brief['topic']}

Per-panel content (each panel is square, {panel_size_px}×{panel_size_px}):

{panels_block}
{('' if empty == 0 else f'{empty} unused cell(s): solid background only, no text, no photo, no decoration.')}

DESIGN — your call:
- You decide the entire design system: background color, typography, accent treatment, layout within each panel, photo cutout style, texture/pattern.
- Make ONE strong cohesive choice and apply it CONSISTENTLY to every panel.
- Bold, share-worthy, instantly recognizable as organic editorial content. Not a paid ad.
- Reference carousels are attached as quality / vibe references — match that level of design but make ORIGINAL creative choices. DO NOT copy them.

BRAND + UNIQUENESS:
- The carousel should feel SPECIFIC to Invitfull — not generic baby-shower content. Lean into the brand's color family (from the brand DNA references) confidently as the dominant palette.
- Make ONE distinctive design move — a signature accent color, an unusual texture, a custom number badge treatment, a specific typographic detail — that makes this carousel identifiable as Invitfull's even without the logo.
- AVOID the generic Pinterest-baby-shower cream-and-blush + script-font + soft-floral aesthetic. Be more confident, more designed, more editorial.
- Keep the cleanliness: generous whitespace, single focal point per panel, no clutter, no badges/stickers/decorations beyond the one signature move.

Hard constraints:
- Each panel is 1:1 square. Single focal point per panel.
- Render the headline (and body where present) in each panel, exactly as quoted.
- The Invitfull logo (from the brand references) appears ONLY on panel {n} (the outro), small and tasteful. Panels 1..{n-1} contain NO logo.
- No floating CTA buttons in panels 1..{n-1}.
- No watermarks. No "logo" text labels.
- Use REAL photorealistic photography (or cutouts of it). NEVER cartoon/illustrated characters.
- Do NOT show printed cards, books, or physical mail. The product is digital.
- No instruction or label text from this prompt should appear in the output (no "PANEL 1", "Headline:", etc.)."""


async def stage2_render_grid(
    client: AsyncOpenAI,
    *,
    reasoning_model: str,
    image_model: str,
    brief: dict,
    n: int,
    cfg: dict,
    style_refs_dir: Path | None,
    enable_brand_refs: bool = True,
    design_spec_path: Path | None = None,
) -> bytes:
    cols, rows, _, _ = panel_layout_description(n)
    grid_w = cols * PANEL_NATIVE_PX
    grid_h = rows * PANEL_NATIVE_PX

    user_content: list[dict] = []
    user_content.extend(style_reference_blocks(style_refs_dir))
    user_content.extend(brand_dna_blocks(cfg, n, enabled=enable_brand_refs,
                                          design_spec_path=design_spec_path))
    user_content.append({"type": "input_text", "text": build_render_prompt(brief, n)})

    response = await client.responses.create(
        model=reasoning_model,
        input=[{"role": "user", "content": user_content}],
        tools=[{
            "type": "image_generation",
            "model": image_model,
            "size": f"{grid_w}x{grid_h}",
            "quality": "high",
        }],
        tool_choice={"type": "image_generation"},
    )

    for item in response.output:
        if getattr(item, "type", None) == "image_generation_call":
            b64 = getattr(item, "result", None)
            if b64:
                return base64.b64decode(b64)
    raise RuntimeError("Stage 2 produced no image")


async def stage3_render_slide_native(
    client: AsyncOpenAI,
    *,
    reasoning_model: str,
    image_model: str,
    grid_bytes: bytes,
    brief: dict,
    slide_index: int,
    cfg: dict,
    enable_brand_refs: bool = True,
    design_spec_path: Path | None = None,
) -> bytes:
    """Pass 2: re-render a single slide at native 1024×1024, using the full
    grid composite as the visual design anchor + brand DNA references for
    direct palette/typographic ground truth."""
    slide = brief["slides"][slide_index]
    n = len(brief["slides"])
    cols, rows, _, _ = panel_layout_description(n)
    col, row = (slide["n"] - 1) % cols, (slide["n"] - 1) // cols

    eyebrow_line = (
        f'Eyebrow tag (small label rendered above the headline): "{slide["eyebrow"]}"\n- '
        if slide.get("eyebrow") else ""
    )
    body_line = (
        f'Body (render this exact text, smaller, beneath the headline): "{slide["body"]}"'
        if slide["body"] else "Body: (none)"
    )

    prompt = f"""You are re-rendering a SINGLE panel from the carousel grid attached.

The full {cols}×{rows} grid composite is provided above as the design anchor. Look specifically at PANEL {slide['n']} — located at column {col + 1}, row {row + 1} (counting from top-left, 1-indexed). Reproduce that panel EXACTLY at native resolution.

Match the design system of the grid:
- Same background color
- Same typography (font family, weight, size hierarchy)
- Same accent colors and treatment
- Same number badge style and placement
- Same photo placement / cutout / framing style
- Same overall composition for this specific panel

Render this panel ONLY (output is one square image, NOT a grid):
- {eyebrow_line}Headline (render this exact text, prominently): "{slide['label']}"
- {body_line}
- Photo subject: {slide['photo_subject']}
- Role: {slide['role']}

Hard constraints:
- Output is 1:1 square, single panel — never a grid.
- Render the headline (and body where present) exactly as quoted.
- Match the grid's design EXACTLY. Do not introduce new design elements.
- No instruction text from this prompt should appear in the output.
"""

    user_content: list[dict] = []
    user_content.extend(brand_dna_blocks(cfg, n, enabled=enable_brand_refs,
                                          design_spec_path=design_spec_path))
    user_content.append({"type": "input_text", "text":
        "Carousel grid composite — design anchor. Reproduce the requested panel "
        "from this grid at native resolution. The brand DNA references above are "
        "the ground truth for accent color and typographic energy — match them:"})
    user_content.append({"type": "input_image", "image_url": encode_bytes(grid_bytes)})
    user_content.append({"type": "input_text", "text": prompt})

    response = await client.responses.create(
        model=reasoning_model,
        input=[{"role": "user", "content": user_content}],
        tools=[{
            "type": "image_generation",
            "model": image_model,
            "size": "1024x1024",
            "quality": "high",
        }],
        tool_choice={"type": "image_generation"},
    )

    for item in response.output:
        if getattr(item, "type", None) == "image_generation_call":
            b64 = getattr(item, "result", None)
            if b64:
                return base64.b64decode(b64)
    raise RuntimeError(f"Pass 2 produced no image for slide {slide_index + 1}")


def resize_to_meta(png_bytes: bytes) -> bytes:
    img = Image.open(BytesIO(png_bytes)).convert("RGB")
    if img.size != (META_SLIDE_PX, META_SLIDE_PX):
        img = img.resize((META_SLIDE_PX, META_SLIDE_PX), Image.LANCZOS)
    out = BytesIO()
    img.save(out, "PNG", optimize=True)
    return out.getvalue()


def split_grid_to_slides(grid_bytes: bytes, n: int) -> list[bytes]:
    cols, rows, _, _ = panel_layout_description(n)
    img = Image.open(BytesIO(grid_bytes)).convert("RGB")
    panel_w = img.width // cols
    panel_h = img.height // rows
    out: list[bytes] = []
    for i in range(n):
        col, row = i % cols, i // cols
        box = (col * panel_w, row * panel_h, (col + 1) * panel_w, (row + 1) * panel_h)
        panel = img.crop(box).resize((META_SLIDE_PX, META_SLIDE_PX), Image.LANCZOS)
        buf = BytesIO()
        panel.save(buf, "PNG", optimize=True)
        out.append(buf.getvalue())
    return out


# ───────────────────────────── orchestration ─────────────────────────────

async def run(args):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set in .env")
        sys.exit(1)

    if args.slides not in GRID_LAYOUTS:
        print(f"ERROR: --slides must be one of {sorted(GRID_LAYOUTS)}")
        sys.exit(1)

    cfg = SEGMENT_CONFIG[args.segment]
    # System instruction is only used when running Stage 1 (gpt-5.5 brief).
    # If a brief is loaded directly via --brief, Stage 1 is skipped.
    system_instruction = ""
    if not args.brief:
        si_path = cfg["carousel_si_file"]
        if not si_path.exists():
            print(f"ERROR: carousel system instruction not found: {si_path}")
            sys.exit(1)
        system_instruction = si_path.read_text()

    style_refs = Path(args.style_refs) if args.style_refs else DEFAULT_STYLE_REFS_DIR
    if not style_refs.exists():
        style_refs = None

    if args.no_design_spec:
        design_spec_path = None
    else:
        design_spec_path = Path(args.design_spec) if args.design_spec else DEFAULT_DESIGN_SPEC
        if not design_spec_path.exists():
            print(f"WARN: design spec not found at {design_spec_path} — proceeding without it")
            design_spec_path = None

    # If --brief provided, load it instead of generating one
    preloaded_brief: dict | None = None
    if args.brief:
        brief_path = Path(args.brief)
        if not brief_path.exists():
            print(f"ERROR: brief file not found: {brief_path}")
            sys.exit(1)
        preloaded_brief = json.loads(brief_path.read_text())
        if len(preloaded_brief["slides"]) != args.slides:
            print(f"ERROR: --brief has {len(preloaded_brief['slides'])} slides, "
                  f"--slides says {args.slides}")
            sys.exit(1)
        topic_for_dir = preloaded_brief.get("topic", "manual_brief")
    else:
        topic_for_dir = args.topic

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    topic_slug = (
        "".join(c if c.isalnum() else "_" for c in topic_for_dir.lower())[:50].strip("_")
    )
    run_dir = cfg["output_root"] / f"carousel_{topic_slug}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Segment: {args.segment}")
    print(f"Topic: {args.topic if not preloaded_brief else preloaded_brief.get('topic', '(from --brief)')}")
    print(f"Slides: {args.slides} ({GRID_LAYOUTS[args.slides][0]}×{GRID_LAYOUTS[args.slides][1]} grid)")
    print(f"Brief source: {'--brief ' + str(args.brief) if preloaded_brief else 'gpt-5.5 stage 1'}")
    print(f"Brand DNA refs: {'OFF' if args.no_brand_refs else 'ON'}")
    print(f"Content model: {args.content_model}")
    print(f"Reasoning model: {args.reasoning_model}")
    print(f"Image model: {args.image_model}")
    print(f"Style refs: {style_refs if style_refs else '(none)'}")
    print(f"Design spec: {design_spec_path if design_spec_path else '(none)'}")
    print(f"Output dir: {run_dir}")
    print("---")

    # max_retries=0: SDK default of 2 silently restarts long-running image
    # gen calls on transient errors, causing 40+ min stacked hangs.
    # timeout=900s: hard 15-min budget per request (Pass 1 normal runtime is
    # ~3-5 min at the smaller grid size).
    client = AsyncOpenAI(api_key=api_key, timeout=900.0, max_retries=0)

    # Stage 1 (skipped if --brief provided)
    if preloaded_brief:
        brief = preloaded_brief
        (run_dir / "brief.json").write_text(json.dumps(brief, indent=2))
        print("Stage 1: SKIPPED (using --brief)")
        print(f"  topic: {brief.get('topic', '(none)')}")
        for s in brief["slides"]:
            body_preview = f' / "{s["body"][:60]}"' if s.get("body") else ""
            ey = f"[{s['eyebrow']}] " if s.get("eyebrow") else ""
            print(f"  s{s['n']}({s['role']}): {ey}\"{s['label']}\"{body_preview}")
    else:
        print("Stage 1: writing carousel brief...")
        brief = await stage1_brief(
            client,
            content_model=args.content_model,
            system_instruction=system_instruction,
            topic=args.topic,
            n_slides=args.slides,
        )
        (run_dir / "brief.json").write_text(json.dumps(brief, indent=2))
        print(f"  topic: {brief['topic']}")
        for s in brief["slides"]:
            body_preview = f' / "{s["body"][:60]}"' if s["body"] else ""
            print(f"  s{s['n']}({s['role']}): \"{s['label']}\"{body_preview}")
    print("---")

    # Pass 1: grid (model commits to one design system across all panels)
    print("Pass 1: rendering panel grid (design system anchor)...")
    grid_bytes = await stage2_render_grid(
        client,
        reasoning_model=args.reasoning_model,
        image_model=args.image_model,
        brief=brief,
        n=args.slides,
        cfg=cfg,
        style_refs_dir=style_refs,
        enable_brand_refs=not args.no_brand_refs,
        design_spec_path=design_spec_path,
    )
    (run_dir / "_grid.png").write_bytes(grid_bytes)
    print("  grid rendered.")

    # Save cropped previews (low-res panels straight from the grid) for reference
    preview_dir = run_dir / "_grid_previews"
    preview_dir.mkdir(exist_ok=True)
    for i, sb in enumerate(split_grid_to_slides(grid_bytes, args.slides), 1):
        (preview_dir / f"slide_{i:02d}.png").write_bytes(sb)
    print(f"  cropped previews saved to {preview_dir.name}/")

    # Pass 2: per-slide native re-render using the grid as visual anchor
    print(f"Pass 2: re-rendering {args.slides} slides natively at 1024² "
          f"(concurrency={args.concurrency})...")
    sem = asyncio.Semaphore(args.concurrency)

    async def render_one(idx: int):
        async with sem:
            try:
                data = await stage3_render_slide_native(
                    client,
                    reasoning_model=args.reasoning_model,
                    image_model=args.image_model,
                    grid_bytes=grid_bytes,
                    brief=brief,
                    slide_index=idx,
                    cfg=cfg,
                    enable_brand_refs=not args.no_brand_refs,
                    design_spec_path=design_spec_path,
                )
                (run_dir / f"slide_{idx + 1:02d}.png").write_bytes(resize_to_meta(data))
                return f"OK slide {idx + 1}"
            except Exception as e:
                return f"FAIL slide {idx + 1}: {type(e).__name__}: {e}"

    results = await asyncio.gather(*(render_one(i) for i in range(args.slides)))
    for r in results:
        print(f"  {r}")

    print(f"\nDone. {run_dir}")


def main():
    p = argparse.ArgumentParser(description="Generate a Meta carousel via OpenAI panel-grid generation")
    p.add_argument("--segment", required=True, choices=list(SEGMENT_CONFIG))
    p.add_argument("--topic", default="",
                   help="Listicle premise (required unless --brief). e.g. '9 baby shower trends taking over 2026'")
    p.add_argument("--slides", required=True, type=int,
                   help=f"Number of slides. Must be one of: {sorted(GRID_LAYOUTS)}")
    p.add_argument("--brief", default=None,
                   help="Path to a pre-written brief JSON. When set, Stage 1 is skipped.")
    p.add_argument("--no-brand-refs", action="store_true",
                   help="Skip passing Invitfull brand DNA references (use for non-Invitfull brands).")
    p.add_argument("--content-model", default=DEFAULT_CONTENT_MODEL)
    p.add_argument("--reasoning-model", default=DEFAULT_REASONING_MODEL)
    p.add_argument("--image-model", default=DEFAULT_IMAGE_MODEL)
    p.add_argument("--style-refs", default=None,
                   help=f"Directory of example carousels for visual style references. "
                        f"Default: {DEFAULT_STYLE_REFS_DIR}")
    p.add_argument("--design-spec", default=None,
                   help=f"Path to a markdown design system spec to inject into "
                        f"every render call. Default: {DEFAULT_DESIGN_SPEC}")
    p.add_argument("--no-design-spec", action="store_true",
                   help="Skip the design spec markdown injection.")
    p.add_argument("--concurrency", type=int, default=16,
                   help="Max parallel per-slide native re-renders in Pass 2 "
                        "(default 16 = render all panels simultaneously for any supported slide count)")
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
