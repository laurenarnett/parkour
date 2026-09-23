#!/usr/bin/env python3
"""Detect whether street parking spots are free in Reolink camera snapshots.

The camera uploads JPEGs over FTP into dated folders, e.g.
    /home/cameraftp/uploads/2025/05/20/Reolink Argus 3 Pro 1_00_20250520180259.jpg

Usage:
    parkour.py check IMAGE        analyze one image and print spot status
    parkour.py watch              analyze new uploads as they arrive
    parkour.py grid IMAGE         draw a coordinate grid + configured spots,
                                  for figuring out spot polygon coordinates
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
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
    cfg.setdefault("output_dir", "/home/cameraftp/parkour")
    cfg.setdefault("model", "yolov8n.pt")
    cfg.setdefault("confidence", 0.3)
    cfg.setdefault("occupied_threshold", 0.3)
    cfg.setdefault("poll_seconds", 2)
    if not cfg.get("spots"):
        sys.exit(f"{path}: define at least one entry in \"spots\"")
    return cfg


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
    """Fraction of the spot polygon covered by the lower half of any vehicle box.

    Only the lower half of each box is used: with the camera looking down the
    street, the top of a tall vehicle overlaps spots behind it, but the part
    touching the ground is what actually sits in a spot.
    """
    h, w = img_shape[:2]
    spot = np.zeros((h, w), np.uint8)
    cv2.fillPoly(spot, [np.array(polygon, np.int32)], 1)
    spot_area = int(spot.sum())
    if spot_area == 0:
        return 0.0

    cars = np.zeros((h, w), np.uint8)
    for x1, y1, x2, y2, *_ in vehicles:
        cars[(y1 + y2) // 2 : y2, x1:x2] = 1
    return float((spot & cars).sum()) / spot_area


def analyze(img, cfg, detector):
    vehicles = detector.vehicles(img)
    spots = []
    for spot in cfg["spots"]:
        overlap = spot_occupancy(img.shape, spot["polygon"], vehicles)
        spots.append({
            "name": spot["name"],
            "occupied": overlap >= cfg["occupied_threshold"],
            "overlap": round(overlap, 3),
        })
    return vehicles, spots


def annotate(img, vehicles, spots, cfg):
    out = img.copy()
    for x1, y1, x2, y2, label, conf in vehicles:
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 160, 0), 2)
        cv2.putText(out, f"{label} {conf:.2f}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 160, 0), 2)
    for spot_cfg, status in zip(cfg["spots"], spots):
        pts = np.array(spot_cfg["polygon"], np.int32)
        color = (0, 0, 255) if status["occupied"] else (0, 200, 0)
        overlay = out.copy()
        cv2.fillPoly(overlay, [pts], color)
        out = cv2.addWeighted(overlay, 0.3, out, 0.7, 0)
        cv2.polylines(out, [pts], True, color, 2)
        word = "TAKEN" if status["occupied"] else "FREE"
        x, y = pts.min(axis=0)
        cv2.putText(out, f"{status['name']}: {word}", (int(x), int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return out


def process(path, cfg, detector):
    img = cv2.imread(str(path))
    if img is None:
        log.warning("could not read %s", path)
        return None
    vehicles, spots = analyze(img, cfg, detector)
    status = {
        "image": str(path),
        "analyzed_at": datetime.now().isoformat(timespec="seconds"),
        "any_free": any(not s["occupied"] for s in spots),
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
        f"{s['name']}={'TAKEN' if s['occupied'] else 'FREE'}({s['overlap']:.0%})"
        for s in spots
    )
    log.info("%s: %s", Path(path).name, summary)
    return status


def list_images(root):
    return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in (".jpg", ".jpeg"))


def watch(cfg, detector, backfill):
    root = cfg["uploads_dir"]
    seen = set() if backfill else set(list_images(root))
    log.info("watching %s (%d existing images skipped)", root, len(seen))
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
                process(path, cfg, detector)
            except Exception:
                log.exception("failed to process %s", path)
        time.sleep(cfg["poll_seconds"])


def grid(image_path, cfg, out_path):
    img = cv2.imread(str(image_path))
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
    p = sub.add_parser("grid")
    p.add_argument("image")
    p.add_argument("-o", "--out", default="grid.jpg")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config(args.config)

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
