from PIL import Image
from pathlib import Path
import numpy as np

folder = Path(r"A:\AITEX_Dataset\Mask_images")

for pattern in ["0044*", "0097*"]:
    for file in sorted(folder.glob(pattern)):
        with Image.open(file) as img:
            arr = np.array(img.convert("L"))

        values = np.unique(arr)
        nonzero = np.count_nonzero(arr)

        print("\nFile:", file.name)
        print("Unique pixel values:", values[:20])
        print("Non-zero pixels:", nonzero)
        print("Defect percentage:", round(nonzero / arr.size * 100, 4), "%")