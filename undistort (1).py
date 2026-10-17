"""Step 2: fix the lens bending in an image and save a before/after picture.

Run:  python undistort.py --image images/test/photo.jpg
"""
import argparse
import os

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser(description="Undistort an image using saved calibration")
    ap.add_argument("--image", required=True)
    ap.add_argument("--calib", default="results/calibration.npz")
    ap.add_argument("--out", default="results/before_after.png")
    args = ap.parse_args()

    data = np.load(args.calib)
    K, D = data["camera_matrix"], data["dist_coeffs"]
    calib_size = tuple(int(v) for v in data["image_size"])

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Cannot read {args.image}")
    h, w = img.shape[:2]
    if (w, h) != calib_size:
        raise SystemExit(f"Image is {(w, h)} but calibration used {calib_size}. "
                         "Use a photo from the same camera and the same resolution.")

    # alpha=0 crops away the empty black edges after correction
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0, (w, h))
    fixed = cv2.undistort(img, K, D, None, new_K)
    x, y, rw, rh = roi
    if rw > 0 and rh > 0:
        fixed = fixed[y:y + rh, x:x + rw]
        fixed = cv2.resize(fixed, (w, h))

    before, after = img.copy(), fixed.copy()
    cv2.putText(before, "BEFORE", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 4)
    cv2.putText(after, "AFTER", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 160, 0), 4)
    combo = np.hstack([before, after])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cv2.imwrite(args.out, combo)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
