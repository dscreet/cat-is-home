import logging
import os
import subprocess
import time
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

import requests
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

INTERVAL = 30
CAT_ID = 15

LOGS_DIR = Path("logs")
ALL_DIR = Path("captures/all")
CATS_DIR = Path("captures/cats")

DISCORD_WEBHOOK_URL = os.environ["DISCORD_WEBHOOK_URL"]

model = YOLO("yolo11s.pt")

logger = logging.getLogger(__name__)


# new log every day
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            TimedRotatingFileHandler(
                LOGS_DIR / "main.log", when="midnight", backupCount=7, encoding="utf-8"
            ),
            logging.StreamHandler(),
        ],
    )


def capture_image(image_path):
    result = subprocess.run(
        ["fswebcam", "-r", "640x480", "--no-banner", str(image_path)],
        capture_output=True,
        text=True,
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
            cat_image_path = CATS_DIR / f"{timestamp}.jpg"
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
    setup_logging()
    logger.info("starting cat surveilance program...")
    while True:
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            image_path = ALL_DIR / f"{timestamp}.jpg"

            # skip detection and notification if image capture fails
            if not capture_image(image_path):
                time.sleep(INTERVAL)
                continue

            cat_image_path = detect_cat(image_path, timestamp)

            if cat_image_path:
                notify_discord(cat_image_path, timestamp)

        except Exception as e:
            logger.exception(f"unexpected error: {e}")

        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
