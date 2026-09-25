#!/usr/bin/env python3
"""Detect whether street parking spots are free in Reolink camera snapshots.

The camera uploads JPEGs over FTP into dated folders, e.g.
    /home/cameraftp/uploads/2025/05/20/Reolink Argus 3 Pro 1_00_20250520180259.jpg

Usage:
    parkour.py check IMAGE        analyze one image and print spot status
    parkour.py watch              analyze new uploads as they arrive, and
                                  notify NTFY_TOPIC when a spot frees up
    parkour.py test-notify        send a test notification
    parkour.py grid IMAGE         draw a coordinate grid + configured spots,
                                  for figuring out spot polygon coordinates
"""

import argparse
import json
import logging
import os
import sys
import time
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("parkour")

# COCO class ids that count as a parked vehicle.
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

DEFAULT_CONFIG = Path(__file__).with_name("config.json")


def load_config(path):
    with open(path) as f:
        cfg = json.load(f)
    cfg.setdefault("uploads_dir", "/home/cameraftp/uploads")
    cfg.setdefault("output_dir", str(Path(__file__).with_name("output")))
    cfg.setdefault("model", "yolov8s.pt")
    cfg.setdefault("confidence", 0.25)
    cfg.setdefault("occupied_threshold", 0.2)
    cfg.setdefault("hidden_threshold", 0.8)
    cfg.setdefault("poll_seconds", 2)
    cfg.setdefault("ignore", [])
    cfg.setdefault("frame_size", [896, 512])
    cfg.setdefault("ntfy_server", "https://ntfy.sh")
    cfg.setdefault("notify_confirm", 1)
    cfg.setdefault("notify_min_hours", 24)
    cfg["street_cleaning"] = {side: [parse_window(w) for w in windows]
                              for side, windows in cfg.get("street_cleaning", {}).items()}
    if not cfg.get("spots"):
        sys.exit(f"{path}: define at least one entry in \"spots\"")
    for spot in cfg["spots"]:
        side = spot.get("side")
        if side is not None and side not in cfg["street_cleaning"]:
            sys.exit(f"{path}: spot {spot['name']} has side {side!r} with no street_cleaning entry")
    return cfg


DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


def parse_window(text):
    """Parse a street cleaning window like "Tue 11:00-12:30" into (weekday, start, end)."""
    day, times = text.split()
    start, end = (datetime.strptime(t, "%H:%M").time() for t in times.split("-"))
    return DAYS.index(day[:3].lower()), start, end


def legal_until(windows, now):
    """When a car parked now must move for street cleaning (now if cleaning is underway).

    Returns None when the side has no cleaning windows.
    """
    for offset in range(8):
        day = now.date() + timedelta(days=offset)
        starts = []
        for weekday, start, end in windows:
            if day.weekday() != weekday:
                continue
            start_at, end_at = datetime.combine(day, start), datetime.combine(day, end)
            if start_at <= now < end_at:
                return now
            if start_at > now:
                starts.append(start_at)
        if starts:
            return min(starts)
    return None


class Detector:
    def __init__(self, model_path, confidence):
        # Imported lazily so `grid` works without ultralytics installed.
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.confidence = confidence

    def vehicles(self, img):
        """Return [(x1, y1, x2, y2, label, conf), ...] for vehicles in img."""
        result = self.model(img, conf=self.confidence, verbose=False)[0]
        found = []
        for box in result.boxes:
            cls = int(box.cls)
            if cls in VEHICLE_CLASSES:
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                found.append((x1, y1, x2, y2, VEHICLE_CLASSES[cls], float(box.conf)))
        return found


def spot_occupancy(img_shape, polygon, vehicles):
    """Return (footprint, full): fractions of the spot polygon covered by vehicles.

    footprint uses only the lower half of each box: with the camera looking
    down the street, the top of a tall vehicle overlaps spots behind it, but
    the part touching the ground is what actually sits in a spot. full uses
    the whole box, which tells us when a spot is blocked from view.
    """
    h, w = img_shape[:2]
    spot = np.zeros((h, w), np.uint8)
    cv2.fillPoly(spot, [np.array(polygon, np.int32)], 1)
    spot_area = int(spot.sum())
    if spot_area == 0:
        return 0.0, 0.0

    lower = np.zeros((h, w), np.uint8)
    whole = np.zeros((h, w), np.uint8)
    for x1, y1, x2, y2, *_ in vehicles:
        lower[(y1 + y2) // 2 : y2, x1:x2] = 1
        whole[y1:y2, x1:x2] = 1
    return (float((spot & lower).sum()) / spot_area,
            float((spot & whole).sum()) / spot_area)


def outside_ignore_zones(vehicles, zones):
    """Drop detections centered in an ignore zone (e.g. a trash bin YOLO calls a car)."""
    kept = []
    for v in vehicles:
        center = ((v[0] + v[2]) / 2, (v[1] + v[3]) / 2)
        if not any(cv2.pointPolygonTest(np.array(z, np.float32), center, False) >= 0 for z in zones):
            kept.append(v)
    return kept


def analyze(img, cfg, detector):
    vehicles = outside_ignore_zones(detector.vehicles(img), cfg["ignore"])
    spots = []
    for spot in cfg["spots"]:
        footprint, full = spot_occupancy(img.shape, spot["polygon"], vehicles)
        if footprint >= cfg["occupied_threshold"]:
            state = "taken"
        elif full >= cfg["hidden_threshold"]:
            # Covered by the top of a vehicle in front (e.g. a double-parked
            # truck), so we can't see whether anything is parked there.
            state = "hidden"
        else:
            state = "free"
        spots.append({"name": spot["name"], "status": state, "overlap": round(footprint, 3)})
    return vehicles, spots


STATUS_COLORS = {"taken": (0, 0, 255), "free": (0, 200, 0), "hidden": (0, 165, 255)}


def annotate(img, vehicles, spots, cfg):
    out = img.copy()
    for x1, y1, x2, y2, label, conf in vehicles:
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 160, 0), 2)
        cv2.putText(out, f"{label} {conf:.2f}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 160, 0), 2)
    for spot_cfg, status in zip(cfg["spots"], spots):
        pts = np.array(spot_cfg["polygon"], np.int32)
        color = STATUS_COLORS[status["status"]]
        overlay = out.copy()
        cv2.fillPoly(overlay, [pts], color)
        out = cv2.addWeighted(overlay, 0.3, out, 0.7, 0)
        cv2.polylines(out, [pts], True, color, 2)
        x, y = pts.min(axis=0)
        cv2.putText(out, f"{status['name']}: {status['status'].upper()}", (int(x), int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return out


def load_image(path, cfg):
    """Read an image, resized to the frame size the spot polygons were drawn on."""
    img = cv2.imread(str(path))
    w, h = cfg["frame_size"]
    if img is not None and img.shape[:2] != (h, w):
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return img


def process(path, cfg, detector):
    img = load_image(path, cfg)
    if img is None:
        log.warning("could not read %s", path)
        return None
    vehicles, spots = analyze(img, cfg, detector)
    status = {
        "image": str(path),
        "analyzed_at": datetime.now().isoformat(timespec="seconds"),
        "any_free": any(s["status"] == "free" for s in spots),
        "spots": spots,
    }

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "latest.jpg"), annotate(img, vehicles, spots, cfg))
    tmp = out_dir / "latest.json.tmp"
    tmp.write_text(json.dumps(status, indent=2))
    tmp.replace(out_dir / "latest.json")
    with open(out_dir / "history.jsonl", "a") as f:
        f.write(json.dumps(status) + "\n")

    summary = ", ".join(
        f"{s['name']}={s['status'].upper()}({s['overlap']:.0%})"
        for s in spots
    )
    log.info("%s: %s", Path(path).name, summary)
    return status


class Notifier:
    """Send a phone notification via ntfy when a spot goes from taken to free.

    The topic comes from the NTFY_TOPIC environment variable rather than
    config.json: anyone who knows an ntfy.sh topic can read it, and the
    notifications include photos of the street.
    """

    def __init__(self, cfg):
        self.topic = os.environ.get("NTFY_TOPIC")
        self.server = cfg["ntfy_server"].rstrip("/")
        self.confirm = cfg["notify_confirm"]
        self.min_hours = cfg["notify_min_hours"]
        self.cleaning = cfg["street_cleaning"]
        self.sides = {spot["name"]: spot.get("side") for spot in cfg["spots"]}
        self.state = {}        # spot name -> last confirmed "taken" / "free"
        self.free_streak = {}  # spot name -> consecutive free readings
        if not self.topic:
            log.warning("NTFY_TOPIC not set; notifications disabled")

    def update(self, status, image_path, now=None):
        opened = []
        for spot in status["spots"]:
            name = spot["name"]
            if spot["status"] == "hidden":
                continue  # can't see it, so keep what we knew before
            if spot["status"] == "taken":
                self.state[name] = "taken"
                self.free_streak[name] = 0
                continue
            self.free_streak[name] = self.free_streak.get(name, 0) + 1
            if self.free_streak[name] >= self.confirm:
                # Only a taken -> free change alerts, so restarting the
                # watcher doesn't announce spots that were already free.
                if self.state.get(name) == "taken":
                    opened.append(name)
                self.state[name] = "free"
        now = now or datetime.now()
        worth_it = []  # (hours legal, message line)
        for name in opened:
            side = self.sides.get(name)
            until = legal_until(self.cleaning[side], now) if side else None
            if until is None:
                worth_it.append((float("inf"), f"{name} is free"))
                continue
            hours = (until - now).total_seconds() / 3600
            if hours < self.min_hours:
                why = "street cleaning in progress" if hours == 0 else f"must move by {until:%a %H:%M}"
                log.info("not notifying %s: free but %s", name, why)
                continue
            left = f"{hours / 24:.0f} days" if hours >= 48 else f"{hours:.0f}h"
            when = until.strftime("%a %-I:%M") + until.strftime("%p").lower()
            worth_it.append((hours, f"{name} is free - good until {when} ({left})"))
        if worth_it:
            worth_it.sort(reverse=True)
            self.send("; ".join(line for _, line in worth_it), image_path)
        return [line for _, line in worth_it]

    def send(self, message, image_path=None):
        if not self.topic:
            return
        data = Path(image_path).read_bytes() if image_path else message.encode()
        headers = {"Title": "Parking spot open", "Tags": "car"}
        if image_path:
            headers.update({"Message": message, "Filename": "parking.jpg"})
        req = urllib.request.Request(f"{self.server}/{self.topic}", data=data,
                                     headers=headers, method="PUT")
        try:
            urllib.request.urlopen(req, timeout=20).close()
            log.info("notified: %s", message)
        except Exception:
            log.exception("failed to send notification")


def list_images(root):
    return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in (".jpg", ".jpeg"))


def watch(cfg, detector, backfill):
    root = cfg["uploads_dir"]
    seen = set() if backfill else set(list_images(root))
    log.info("watching %s (%d existing images skipped)", root, len(seen))
    notifier = Notifier(cfg)
    while True:
        for path in list_images(root):
            if path in seen:
                continue
            # FTP may still be writing the file; wait until it stops changing.
            try:
                if time.time() - path.stat().st_mtime < 2:
                    continue
            except FileNotFoundError:
                continue
            seen.add(path)
            try:
                status = process(path, cfg, detector)
                if status:
                    notifier.update(status, Path(cfg["output_dir"]) / "latest.jpg")
            except Exception:
                log.exception("failed to process %s", path)
        time.sleep(cfg["poll_seconds"])


def grid(image_path, cfg, out_path):
    img = load_image(image_path, cfg)
    if img is None:
        sys.exit(f"could not read {image_path}")
    h, w = img.shape[:2]
    step = 100
    for x in range(0, w, step):
        cv2.line(img, (x, 0), (x, h), (255, 255, 255), 1)
        cv2.putText(img, str(x), (x + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    for y in range(0, h, step):
        cv2.line(img, (0, y), (w, y), (255, 255, 255), 1)
        cv2.putText(img, str(y), (2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    for spot in cfg.get("spots", []):
        pts = np.array(spot["polygon"], np.int32)
        cv2.polylines(img, [pts], True, (0, 255, 255), 2)
        cv2.putText(img, spot["name"], tuple(int(v) for v in pts[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    for zone in cfg.get("ignore", []):
        cv2.polylines(img, [np.array(zone, np.int32)], True, (255, 0, 255), 2)
    cv2.imwrite(str(out_path), img)
    print(f"{w}x{h} image, grid written to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--config", default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check")
    p.add_argument("image")
    p = sub.add_parser("watch")
    p.add_argument("--backfill", action="store_true", help="also analyze images already on disk")
    sub.add_parser("test-notify", help="send a test notification to NTFY_TOPIC")
    p = sub.add_parser("grid")
    p.add_argument("image")
    p.add_argument("-o", "--out", default="grid.jpg")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config(args.config)

    if args.cmd == "test-notify":
        notifier = Notifier(cfg)
        if not notifier.topic:
            sys.exit("set NTFY_TOPIC first")
        notifier.send("Test from parkour: notifications are working")
        return

    if args.cmd == "grid":
        grid(args.image, cfg, args.out)
        return

    detector = Detector(cfg["model"], cfg["confidence"])
    if args.cmd == "check":
        status = process(args.image, cfg, detector)
        if status:
            print(json.dumps(status, indent=2))
    else:
        watch(cfg, detector, args.backfill)


if __name__ == "__main__":
    main()
