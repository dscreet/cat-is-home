import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

model = YOLO("yolo11s.pt")

interval = 30
all_dir = Path("captures/all")
cats_dir = Path("captures/cats")
CAT_ID = 15
DISCORD_WEBHOOK_URL = os.environ["DISCORD_WEBHOOK_URL"]


def capture_image(image_path):
    subprocess.run(
        ["fswebcam", "-r", "640x480", "--no-banner", str(image_path)],
        capture_output=True,
    )
    print(f"Saved {image_path}")


def detect_cat(image_path, timestamp):
    result = model(image_path)[0]  # since i am using one image, array size is one
    detected_classes = result.boxes.cls

    if CAT_ID in detected_classes:
        print("cat detected")
        cat_image_path = cats_dir / f"{timestamp}.jpg"
        result.save(filename=str(cat_image_path))
        return cat_image_path

    return None


def notify_discord(cat_image_path, timestamp):
    with open(cat_image_path, "rb") as f:
        requests.post(
            DISCORD_WEBHOOK_URL,
            data={"content": "cat detected"},
            files={"file": (f"{timestamp}.jpg", f, "image/jpeg")},
        )
    print("notification sent")


def main():
    while True:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        image_path = all_dir / f"{timestamp}.jpg"

        capture_image(image_path)

        cat_image_path = detect_cat(image_path, timestamp)

        if cat_image_path:
            notify_discord(cat_image_path, timestamp)

        time.sleep(interval)


if __name__ == "__main__":
    main()
