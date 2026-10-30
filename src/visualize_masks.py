from PIL import Image, ImageDraw
from pathlib import Path

base = Path(r"A:\AITEX_Dataset")
output = Path("outputs")
output.mkdir(exist_ok=True)

pairs = [
    ("0044_019_04.png", "0044_019_04_mask1.png"),
    ("0044_019_04.png", "0044_019_04_mask2.png"),
    ("0097_030_03.png", "0097_030_03_mask1.png"),
    ("0097_030_03.png", "0097_030_03_mask2.png"),
]

panels = []

for image_name, mask_name in pairs:
    with Image.open(base / "Defect_images" / image_name) as im:
        image = im.convert("RGB")

    with Image.open(base / "Mask_images" / mask_name) as m:
        mask = m.getchannel("L") if m.mode == "LA" else m.convert("L")

    bbox = mask.getbbox()

    if bbox is None:
        print("Empty mask:", mask_name)
        continue

    x1, y1, x2, y2 = bbox
    margin = 40

    crop_box = (
        max(0, x1 - margin),
        max(0, y1 - margin),
        min(image.width, x2 + margin + 1),
        min(image.height, y2 + margin + 1)
    )

    img_crop = image.crop(crop_box)
    mask_crop = mask.crop(crop_box)

    overlay = img_crop.copy()
    red = Image.new("RGB", overlay.size, (255, 0, 0))
    overlay.paste(red, mask=mask_crop.point(
        lambda p: 255 if p > 0 else 0
    ))

    overlay = overlay.resize(
        (overlay.width * 5, overlay.height * 5)
    )

    panel = Image.new("RGB", (overlay.width, overlay.height + 30), "white")
    panel.paste(overlay, (0, 30))
    ImageDraw.Draw(panel).text(
        (5, 5), mask_name, fill="black"
    )

    panels.append(panel)

if panels:
    width = max(p.width for p in panels)
    height = sum(p.height for p in panels)

    sheet = Image.new("RGB", (width, height), "white")
    y = 0

    for panel in panels:
        sheet.paste(panel, (0, y))
        y += panel.height

    sheet.save(output / "mask_visual_check.png")
    print("Saved: outputs/mask_visual_check.png")