import logging
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv
from ultralytics import YOLO

# Set up logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

load_dotenv()

model = YOLO("yolo11s.pt")

interval = 30
all_dir = Path("captures/all")
cats_dir = Path("captures/cats")
CAT_ID = 15
DISCORD_WEBHOOK_URL = os.environ["DISCORD_WEBHOOK_URL"]


def capture_image(image_path):
    result = subprocess.run(
        ["fswebcam", "-r", "640x480", "--no-banner", str(image_path)],
        capture_output=True,
        # text=True,
    )
    if result.returncode != 0:
        logger.error(f"image capture failed: {result.stderr}")
        return False
    if not image_path.exists():
        logger.error(f"image was not saved: {image_path}")
        return False
    logger.info(f"saved {image_path}")
    return True


def detect_cat(image_path, timestamp):
    try:
        result = model(image_path)[0]
        detected_classes = result.boxes.cls

        if CAT_ID in detected_classes:
            logger.info("cat detected")
            cat_image_path = cats_dir / f"{timestamp}.jpg"
            result.save(filename=str(cat_image_path))
            return cat_image_path

        return None
    except Exception as e:
        logger.error(f"detection failed: {e}")
        return None


def notify_discord(cat_image_path, timestamp):
    try:
        with open(cat_image_path, "rb") as f:
            response = requests.post(
                DISCORD_WEBHOOK_URL,
                data={"content": "cat detected"},
                files={"file": (f"{timestamp}.jpg", f, "image/jpeg")},
                timeout=30,
            )
            response.raise_for_status()
        logger.info("discord notification sent")
    except requests.RequestException as e:
        logger.error(f"failed to send discord notification: {e}")


def main():
    logger.info("starting cat surveilance program...")
    while True:
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            image_path = all_dir / f"{timestamp}.jpg"

            # skip detection and notification if image capture fails
            if not capture_image(image_path):
                time.sleep(interval)
                continue

            cat_image_path = detect_cat(image_path, timestamp)

            if cat_image_path:
                notify_discord(cat_image_path, timestamp)

        except Exception as e:
            logger.exception(f"unexpected error: {e}")

        time.sleep(interval)


if __name__ == "__main__":
    main()
