# Ad Creative Pipeline

A Python pipeline for generating brand-aligned ad creatives at scale, using OpenAI's `gpt-image-2` (and optionally Google's `gemini-3-pro-image-preview`).

Two output modes:

1. **Single-image batch** (`run_batch_openai.py`) — given a folder of inspiration images, produces one Invitfull-branded ad per inspiration. Useful for recreating proven competitor ads.
2. **Editorial carousels** (`run_carousel_openai.py`) — given a topic, produces a multi-slide Instagram/Facebook carousel laid out as viral-feeling editorial content (number-driven listicle hook → concept slides → soft brand outro).

Both pipelines respect a per-segment system instruction file and a per-segment set of reference assets (logo, product UI screenshots, invitation examples) for brand fidelity.

---

## Setup

Requires Python 3.9+.

```bash
git clone <this-repo>
cd ad-creative-pipeline

# Create a venv (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install deps
pip install -r requirements.txt

# Configure secrets
cp .env.example .env
# then edit .env and paste your OPENAI_API_KEY
```

The OpenAI key needs **Tier 2 or higher** for fast carousels (Tier 1 = 5 images/min, which serializes Pass 2 calls). On Tier 1 a 9-slide carousel takes ~14 min; on Tier 2+ it's closer to 7 min.

---

## Project layout

The pipeline expects a per-segment directory under the repo root. Each segment has its own brand assets and system instructions.

```
your-project-root/
├── .env                                 # OPENAI_API_KEY (gitignored)
├── scripts/                             # pipeline scripts
├── baby-shower/                         # one segment
│   ├── system_instruction.md            # voice + rules for single-image ads
│   ├── system_instruction_carousel.md   # voice + rules for carousels
│   ├── reference-images/
│   │   ├── logo/logo.png                # brand wordmark
│   │   ├── screenshots/                 # product UI references (3+ recommended)
│   │   └── invitations/                 # optional: example outputs
│   └── output/                          # generated ads land here (gitignored)
├── kids-birthday/                       # another segment, same structure
└── general/                             # generic segment, same structure
```

The three baked-in segment names are `baby_shower`, `kids_birthday`, and `generic`. Add or rename segments by editing the `SEGMENT_CONFIG` dict at the top of each script.

### System instruction files

Examples are in `examples/`:

- `system_instruction.md.example` — for single-image ads. Defines the brand voice, format detection (branded vs authentic), copy rules, and hard constraints. Copy to `<segment>/system_instruction.md` and tune.
- `system_instruction_carousel.md.example` — for carousels. Defines the listicle structure, voice, and the rule that the brand only appears in the outro.

### Reference images

Drop your brand assets into `<segment>/reference-images/`:

- `logo/logo.png` — wordmark, used as a paste-exact reference (never redrawn) and on the carousel outro slide
- `screenshots/` — product UI screenshots — referenced for color/typography DNA, never collaged into the output
- `invitations/` — optional, example real outputs that can be featured in branded ads

These are gitignored by default so they stay local. Don't commit your brand assets unless your repo is private.

### Optional: design system spec

For the carousel pipeline, you can supply a markdown file describing your brand's full design system (palette hex codes, typography, pattern recipes, voice guidelines). Pass it via `--design-spec /path/to/design.md` or place it at the default path that matches your project. The spec text is injected into every render call so the model has the full brand context when picking colors, typography, and composition.

The default path is hardcoded to a personal Desktop location — change `DEFAULT_DESIGN_SPEC` in `scripts/run_carousel_openai.py` to fit your setup, or always pass `--design-spec`.

---

## Quick start

### Generate a single-image batch

Recreate a folder of ad inspirations as Invitfull-branded ads.

```bash
.venv/bin/python scripts/run_batch_openai.py \
  --segment baby_shower \
  --dir /path/to/inspirations
```

What happens:

1. Discovers all `.png/.jpg/.jpeg/.webp` files in `--dir`.
2. For each inspiration, makes one Responses-API call. The call passes the segment's logo + screenshots + invitations + the inspiration as input images, plus the system instruction. The image_generation tool returns one Invitfull-branded ad in the inspiration's aspect ratio.
3. Saves outputs to `<segment>/output/` with descriptive filenames.

Useful flags:

- `--ratios 1:1,9:16` — generate at multiple ratios per inspiration (default: auto-detect from the inspiration)
- `--motivation time_pressure` — appends a focus directive to the system instruction (any phrase works)
- `--concurrency 4` — parallel calls (default 4; raise on Tier 2+)
- `--reasoning-model gpt-5.1` — override the orchestrating multimodal model (default `gpt-4o`)
- `--image-model gpt-image-1.5` — override (default `gpt-image-2`)

### Generate a carousel

Editorial-style multi-slide carousel from a topic.

```bash
.venv/bin/python scripts/run_carousel_openai.py \
  --segment baby_shower \
  --topic "9 baby shower trends taking over 2026" \
  --slides 9
```

What happens:

1. **Stage 1 — content brief.** `gpt-5.5` reads your system instruction and the topic, and returns a JSON-schema-validated brief describing each slide (label, body, photo subject, role).
2. **Pass 1 — design grid.** A single image-generation call produces a composite N-panel grid. The model commits to ONE design system across all panels in this call.
3. **Pass 2 — native re-renders.** All N slides are re-rendered at native 1024² in parallel, with the Pass 1 grid passed back as the visual anchor + the brand DNA references + (optional) design spec on every call.
4. Saves slides + grid composite + brief.json to `<segment>/output/carousel_<topic>_<timestamp>/`.

Useful flags:

- `--slides N` — must be 1–10. Math: `N = 1 hook + (N-2) concepts + 1 outro`. So a "7 mistakes" listicle wants `--slides 9`.
- `--brief /path/to/brief.json` — skip Stage 1 and use a pre-written brief (see schema below). Useful when you have exact copy you want rendered.
- `--no-brand-refs` — skip Invitfull brand DNA references. Use for non-Invitfull brands.
- `--no-design-spec` — skip the design markdown injection.
- `--design-spec /path/to/design.md` — point at a different design spec file.
- `--style-refs /path/to/example-carousels` — directory of example carousels to use as visual energy references (default: a personal Desktop folder; change or pass your own).
- `--concurrency 16` — Pass 2 parallelism (default 16; effective cap = your OpenAI tier's IPM limit).

### Pre-written brief schema

When using `--brief`, the JSON must look like:

```json
{
  "topic": "Your BFF is having a baby shower — here's what you need to know",
  "slides": [
    {
      "n": 1,
      "role": "hook",
      "eyebrow": "BABY SHOWER",
      "label": "Your BFF is having a baby shower. Here's what you need to know.",
      "body": "You're not just a guest. You're the one who makes it special.",
      "photo_subject": "two close women friends, candid, warm light"
    },
    {
      "n": 2,
      "role": "concept",
      "eyebrow": "TIP 1",
      "label": "Offer something specific.",
      "body": "She won't ask. Show up and say \"I'm handling the food.\"",
      "photo_subject": "hands carrying a cake to a friend"
    }
    // ... etc
  ]
}
```

Fields per slide:

- `n` (int) — slide number, 1-indexed
- `role` (string) — `hook` | `concept` | `outro`
- `label` (string) — the big headline rendered on the slide, exact text
- `body` (string) — optional 1-2 sentence editorial subhead, empty `""` for none
- `eyebrow` (string) — optional small label tag rendered above the headline (e.g. `TIP 1`)
- `photo_subject` (string) — 1-3 word description of the photo content, NOT the design

---

## Architecture notes

### Why two passes for carousels

Pass 1 generates a composite N-panel grid in a single call. This forces the model to commit to one design system (palette, typography, layout pattern) across all panels — locking visual continuity natively rather than trying to reconstruct it across N independent calls.

Pass 2 then re-renders each slide at native 1024² resolution, using the Pass 1 grid as a visual anchor. This recovers the per-slide pixel quality that's lost when you crop a 1824² grid into 608² panels.

### Why the panel size is 608

`gpt-image-2` requires both image edges to be divisible by 16, and starts behaving experimentally above ~3.7M total pixels (`2560×1440` is the documented reliability ceiling). `608` is divisible by 16, and 9 panels at `608² = 1824² total` keeps us well under the ceiling. `768²` panels (giving a `2304² total` grid) was the original setting and silently hung in production — it's above the reliability threshold.

### Why `max_retries=0`

The OpenAI SDK's default `max_retries=2` silently restarts long-running image-generation calls on transient timeouts. Stacked with a generous client timeout, this can produce 75-minute hangs with no error visible. We set `max_retries=0` and a hard `timeout=900` per request so failures bubble up immediately.

### Cost per carousel (rough)

- Stage 1 (gpt-5.5 with structured output, ~50K tokens of context): ~$0.05
- Pass 1 (high-quality grid, gpt-image-2, ~3.3M pixels): ~$0.40–$0.80
- Pass 2 (N high-quality 1024² renders): ~$0.20 × N

For a 9-slide carousel: ~$2.30 per run.

---

## Troubleshooting

**"insufficient_quota" / 429** — your OpenAI project is out of credit. Top up at platform.openai.com/account/limits.

**Pass 1 hangs for 10+ minutes** — usually means you exceeded the size reliability ceiling. Check that your panel × cols and panel × rows stay below 2560 and total pixels stay below 3.7M. The current 608px panel size is safe up to a 4×2 grid (2432×1216).

**Body text rendered as "Body…" placeholder** — happens occasionally on slides where the body string is too long for the panel size. Shorten the body or accept the truncation; not a pipeline bug.

**Brand name rendered when `--no-brand-refs` is set** — known issue. The render prompt still mentions the Invitfull name in the logo-placement instruction even when refs are off. Fix is to gate that prompt sentence; not yet patched.

**Cross-slide drift in carousel design** — increase the design spec specificity (more hex codes, more typography rules) or pass example carousels via `--style-refs` to anchor the visual energy.

---

## File map

| Path | Purpose |
|---|---|
| `scripts/run_carousel_openai.py` | OpenAI two-pass carousel pipeline |
| `scripts/run_batch_openai.py` | OpenAI single-image batch (one ad per inspiration) |
| `scripts/run_batch.py` | Original Gemini batch (Nano Banana Pro / `gemini-3-pro-image-preview`) — kept for A/B comparison |
| `scripts/generate_ad.py` | Gemini single-image one-off |
| `examples/system_instruction.md.example` | Example system instruction for single-image ads |
| `examples/system_instruction_carousel.md.example` | Example system instruction for carousels |
| `.env.example` | Environment variable template |
| `requirements.txt` | Python dependencies |

---

## Contributing

When changing the carousel renderer:

- Honor the 16-divisibility constraint on grid dimensions
- Stay under the 2560×1440 reliability ceiling for the grid (3.7M pixels)
- Keep `max_retries=0` on the AsyncOpenAI client
- Test against both 5-slide and 9-slide topics before merging — they exercise different grid layouts (3×2 vs 3×3)

When changing brand voice:

- Update the segment's `system_instruction.md` and `system_instruction_carousel.md`
- Avoid hardcoding brand-specific strings in the Python scripts; the script should stay brand-agnostic
