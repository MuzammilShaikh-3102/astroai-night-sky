from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont
import numpy as np


def annotate_image(image: Image.Image, detections, matches, catalog, constellation_edges, planet_points, matched_star_names) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("arial.ttf", max(14, out.width // 85))
        small = ImageFont.truetype("arial.ttf", max(12, out.width // 105))
    except Exception:
        font = None
        small = None

    # Draw all detected points softly.
    for p in detections:
        draw.ellipse((p.x-3,p.y-3,p.x+3,p.y+3), outline=(255,255,255), width=1)

    name_positions = {}
    for det_idx, cat_idx, dist in matches:
        row = catalog.iloc[cat_idx]
        p = detections[det_idx]
        name = str(row["display_name"])
        name_positions[name] = (p.x, p.y)
        r = 7
        draw.ellipse((p.x-r,p.y-r,p.x+r,p.y+r), outline=(255,215,0), width=3)
        tx, ty = p.x + 8, p.y - 8
        bbox = draw.textbbox((tx,ty), name, font=small)
        draw.rounded_rectangle(bbox, radius=5, fill=(0,0,0), outline=(255,215,0), width=1)
        draw.text((tx,ty), name, fill=(255,255,255), font=small)

    for planet, (x,y,alt) in planet_points.items():
        r=10
        draw.ellipse((x-r,y-r,x+r,y+r), outline=(0,220,255), width=3)
        text=f"{planet}"
        bbox=draw.textbbox((x+10,y-10),text,font=small)
        draw.rounded_rectangle(bbox,radius=5,fill=(0,0,0),outline=(0,220,255),width=1)
        draw.text((x+10,y-10),text,fill=(255,255,255),font=small)

    # Constellation lines only when both endpoint stars were matched.
    for a,b in constellation_edges:
        if a in name_positions and b in name_positions:
            ax,ay=name_positions[a]
            bx,by=name_positions[b]
            draw.line((ax,ay,bx,by), fill=(120,200,255), width=max(2,out.width//500))

    return out
