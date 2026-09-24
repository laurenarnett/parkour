# parkour

Checks whether a street parking spot is free in snapshots that a Reolink Argus 3 Pro uploads over FTP to a Raspberry Pi.

YOLOv8 finds the vehicles in each photo. A spot counts as taken when the bottom half of a vehicle's box covers at least `occupied_threshold` of the spot's polygon.

## Setup (on the Pi)

    git clone <this repo> ~/parkour-app && cd ~/parkour-app
    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

## Mark your spots

1. Draw a pixel grid over one of the camera's photos:

       .venv/bin/python parkour.py grid "/home/cameraftp/uploads/2025/05/20/Reolink Argus 3 Pro 1_00_20250520180259.jpg"

2. Open `grid.jpg` and read off the corners of each parking spot.
3. Put those corners in `config.json` under `spots`.
   Coordinates are in the `frame_size` frame (896×512 by default). Photos of any other size are resized to it first, so the spots still line up if the camera's upload resolution changes.
4. Run `grid` again to check that the yellow outlines line up with the spots.

## Run

    .venv/bin/python parkour.py check "<image>"   # one image
    .venv/bin/python parkour.py watch             # new uploads as they arrive

Results go to `output_dir`:
- `latest.json`: current status, including `any_free`
- `latest.jpg`: the photo with spots marked FREE, TAKEN or HIDDEN (blocked from view, e.g. by a double-parked truck)
- `history.jsonl`: one line per analyzed image

To run it as a service, edit the paths and user in `parkour.service` if needed, then:

    sudo cp parkour.service /etc/systemd/system/ && sudo systemctl enable --now parkour
