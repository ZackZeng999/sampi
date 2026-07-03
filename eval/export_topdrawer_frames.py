from pathlib import Path
import io

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


BASE = Path("/root/autodl-tmp/datasets/physical-intelligence/libero_sam_dim_2/data")
OUT = Path("/root/proj/eval/prompt_mask_audit_first_hit/spatial_topdrawer_video_frames.jpg")
EPISODES = [1080, 1081, 1086, 1094, 1102, 1116, 1120, 1152, 1157, 1167, 1181, 1198]


def episode_path(ep: int) -> Path:
    chunk = "chunk-001" if ep >= 1000 else "chunk-000"
    return BASE / chunk / f"episode_{ep:06d}.parquet"


def decode_image(obj) -> Image.Image:
    if isinstance(obj, dict) and "bytes" in obj:
        return Image.open(io.BytesIO(obj["bytes"])).convert("RGB")
    if isinstance(obj, (bytes, bytearray)):
        return Image.open(io.BytesIO(obj)).convert("RGB")
    return Image.fromarray(np.asarray(obj)).convert("RGB")


def main() -> None:
    cells = []
    labels = []
    for ep in EPISODES[:8]:
        df = pd.read_parquet(episode_path(ep))
        idxs = [0, min(20, len(df) - 1), min(60, len(df) - 1), len(df) - 1]
        for ix in idxs:
            im = decode_image(df.iloc[ix]["image"]).resize((256, 256))
            cells.append(im)
            labels.append(f"ep{ep} f{ix} len{len(df)}")

    cols = 4
    cell_h = 292
    sheet = Image.new("RGB", (cols * 256, ((len(cells) + cols - 1) // cols) * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    for i, im in enumerate(cells):
        x = (i % cols) * 256
        y = (i // cols) * cell_h
        sheet.paste(im, (x, y))
        draw.text((x + 5, y + 262), labels[i], fill=(0, 0, 0))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
