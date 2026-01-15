#!/usr/bin/env python3
"""Convert invoice images into single-page PDFs for the Document AI pipeline.

Default paths (repo-root relative):
- Input:  data/images/
- Output: data/pdfs/

Behavior:
- One PDF per image
- Filename preserved (image.jpg -> image.pdf)
- Converts to RGB before saving
- Skips non-image files
- Prints progress logs

Dependencies: Pillow (PIL)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, UnidentifiedImageError


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def convert_folder(input_dir: Path, output_dir: Path) -> int:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"[WARN] Input folder does not exist: {input_dir}")
        print("       Nothing to convert.")
        return 0

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    candidates = sorted(p for p in input_dir.iterdir() if p.is_file())
    if not candidates:
        print(f"[INFO] No files found in: {input_dir}")
        return 0

    converted = 0
    total = len(candidates)

    print(f"[INFO] Converting files from {input_dir} -> {output_dir}")
    for idx, image_path in enumerate(candidates, start=1):
        suffix = image_path.suffix.lower()

        # Fast skip for obviously non-image extensions; still allow unknowns to be tried.
        if suffix and suffix not in IMAGE_EXTENSIONS:
            print(f"[{idx}/{total}] Skipping non-image file: {image_path.name}")
            continue

        out_path = output_dir / f"{image_path.stem}.pdf"

        try:
            with Image.open(image_path) as img:
                rgb = img.convert("RGB")
                rgb.save(out_path, format="PDF")
            converted += 1
            print(f"[{idx}/{total}] OK  {image_path.name} -> {out_path.name}")
        except UnidentifiedImageError:
            print(f"[{idx}/{total}] Skipping non-image file: {image_path.name}")
        except Exception as exc:  # keep the batch running
            print(f"[{idx}/{total}] FAIL {image_path.name}: {exc}")

    print(f"[DONE] Converted {converted} image(s) to PDF.")
    return converted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert images in data/images/ into one-PDF-per-image in data/pdfs/."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/images"),
        help="Input folder containing JPG/PNG scans (default: data/images)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/pdfs"),
        help="Output folder to write PDFs (default: data/pdfs)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_folder(args.input, args.output)


if __name__ == "__main__":
    main()
