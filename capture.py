import subprocess
import time
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

model = YOLO("yolo11n.pt")

interval = 30
all_dir = Path("captures/all")
cats_dir = Path("captures/cats")
CAT_ID = 15

while True:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    image_path = all_dir / f"{timestamp}.jpg"

    subprocess.run(
        ["fswebcam", "-r", "640x480", "--no-banner", str(image_path)],
        capture_output=True,
    )

    print(f"Saved {image_path}")

    result = model(image_path)[0]  # since i am using one image, array size is one
    detected_classes = result.boxes.cls

    if CAT_ID in detected_classes:
        print("cat detected")
        cat_image_path = cats_dir / f"{timestamp}.jpg"
        result.save(filename=str(cat_image_path))

    time.sleep(interval)
