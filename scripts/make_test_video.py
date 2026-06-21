"""
Build a test video from a folder of star-field frames (default: the KnacksatOrbit
frames). Uses lossless FFV1 so faint stars survive compression -- a lossy codec
dims them and tanks detection.

Usage: python scripts/make_test_video.py [frame_folder] [output.avi] [fps]
       defaults: cache/KnacksatOrbit_frame  outputs/knacksat_orbit.avi  10
"""
import sys
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parent.parent


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "cache" / "KnacksatOrbit_frame"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "outputs" / "knacksat_orbit.avi"
    fps = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0

    frames = sorted(folder.glob("*.png"))
    if not frames:
        sys.exit(f"No .png frames in {folder}")
    h, w = cv2.imread(str(frames[0])).shape[:2]
    out.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"FFV1"), fps, (w, h))
    if not writer.isOpened():
        sys.exit("FFV1 codec unavailable in this OpenCV build")

    for i, f in enumerate(frames, 1):
        writer.write(cv2.imread(str(f)))
        if i % 20 == 0 or i == len(frames):
            print(f"  {i}/{len(frames)} frames written")
    writer.release()
    print(f"[video] wrote {out} ({w}x{h} @ {fps} fps, {len(frames)} frames)")


if __name__ == "__main__":
    main()
