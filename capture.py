import subprocess
import time
from datetime import datetime
from pathlib import Path

interval = 10
output_dir = Path("captures")

while True:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    image_path = output_dir / f"{timestamp}.jpg"

    # will save images only when cat is detected later
    subprocess.run(
        ["fswebcam", "-r", "1280x720", "--no-banner", str(image_path)],
        capture_output=True,
    )

    print(f"Saved {image_path}")

    time.sleep(interval)
