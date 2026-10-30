from PIL import Image
from pathlib import Path
import numpy as np

folder = Path(r"A:\AITEX_Dataset\Mask_images")

files = [
    "0044_019_04_mask1.png",
    "0044_019_04_mask2.png",
    "0097_030_03_mask1.png",
    "0097_030_03_mask2.png"
]

for name in files:
    path = folder / name

    with Image.open(path) as img:
        print("\nFILE:", name)
        print("Mode:", img.mode)
        print("Size:", img.size)

        for channel_name, channel in zip(
            img.getbands(), img.split()
        ):
            arr = np.array(channel)

            print("Channel:", channel_name)
            print("Min:", arr.min(), "Max:", arr.max())
            print("Non-zero pixels:", np.count_nonzero(arr))
            print("Unique values:", np.unique(arr)[:15])