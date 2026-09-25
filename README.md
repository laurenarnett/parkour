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

## Phone notifications

The watcher sends an [ntfy](https://ntfy.sh) notification, with the annotated photo attached, when a spot goes from taken to free.

1. Install the ntfy app on your phone and subscribe to a hard-to-guess topic name. Anyone who knows the name can read the photos.
2. On the Pi, put the topic in `notify.env`. This file is git-ignored, so the name stays out of this public repo:

       echo NTFY_TOPIC=your-topic-name > notify.env

3. Send a test: `set -a; . ./notify.env; .venv/bin/python parkour.py test-notify`

To wait for N free readings in a row before alerting, set `notify_confirm` in `config.json`.

### Street cleaning

Each spot has a `side`, and `street_cleaning` lists each side's cleaning windows (e.g. `"Tue 11:00-12:30"`). Alerts say how long a spot stays legal ("good until Mon 11:00am (4 days)"). Spots you'd have to move out of within `notify_min_hours` (24 by default) don't alert, and neither do spots on a side being cleaned right now. In the last `notify_before_cleaning_ends_minutes` (15 by default) of a side's cleaning window, any spot on that side that a photo shows as free alerts once, e.g. "near-1 is free after cleaning ends at 12:30pm - good until Tue 11:00am (4 days)". A spot emptied for cleaning never counts as a new opening, so without this you'd never hear about it.

Holiday suspensions come from [NYC DOT's alternate side parking calendar](https://www.nyc.gov/html/dot/html/motorist/alternate-side-parking.shtml). It's downloaded once a day, and a copy is cached in `output/`. A suspended cleaning day is skipped when working out how long a spot is good for, and the alert notes it ("Mon 10/12 cleaning suspended"). Last-minute suspensions, e.g. for snow, aren't in the calendar. Set `suspension_calendar_url` to `""` to turn this off.

To run it as a service, edit the paths and user in `parkour.service` if needed, then:

    sudo cp parkour.service /etc/systemd/system/ && sudo systemctl enable --now parkour
