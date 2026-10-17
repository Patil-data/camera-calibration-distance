"""Step 1: find the camera's hidden numbers from chessboard photos.

Run:  python calibrate.py --square-cm 2.5
"""
import argparse
import glob
import os

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser(description="Camera calibration with a chessboard")
    ap.add_argument("--images", default="images/chessboard", help="folder with chessboard photos")
    ap.add_argument("--cols", type=int, default=9, help="inner corners along the width")
    ap.add_argument("--rows", type=int, default=6, help="inner corners along the height")
    ap.add_argument("--square-cm", type=float, required=True, help="size of ONE printed square in cm")
    ap.add_argument("--out", default="results/calibration.npz")
    args = ap.parse_args()

    pattern = (args.cols, args.rows)

    # Where the corners SHOULD be on a perfect flat board (in cm).
    objp = np.zeros((args.rows * args.cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * args.square_cm

    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        paths += glob.glob(os.path.join(args.images, ext))
    paths = sorted(set(paths))
    if not paths:
        raise SystemExit(f"No images found in {args.images}")

    os.makedirs("results/corners", exist_ok=True)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    obj_points, img_points, used = [], [], []
    image_size = None

    for path in paths:
        img = cv2.imread(path)
        if img is None:
            print(f"[skip] cannot read {path}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]  # (width, height)
        if image_size is None:
            image_size = size
        elif size != image_size:
            print(f"[skip] {os.path.basename(path)}: size {size} differs from {image_size}")
            continue

        found, corners = cv2.findChessboardCorners(
            gray, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if not found:
            print(f"[skip] {os.path.basename(path)}: chessboard not found")
            continue

        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        obj_points.append(objp)
        img_points.append(corners)
        used.append(os.path.basename(path))
        cv2.drawChessboardCorners(img, pattern, corners, found)
        cv2.imwrite(os.path.join("results/corners", os.path.basename(path)), img)
        print(f"[ok]   {os.path.basename(path)}")

    n = len(obj_points)
    print(f"\nValid images: {n} / {len(paths)}")
    if n < 10:
        raise SystemExit("Need at least 10 valid images. Take more photos from different angles.")

    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, image_size, None, None)

    # Error check: project the 3D corners back and compare with what we detected.
    per_image = []
    for i in range(n):
        proj, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, D)
        err = cv2.norm(img_points[i], proj, cv2.NORM_L2) / np.sqrt(len(proj))
        per_image.append(err)
    mean_err = float(np.mean(per_image))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, camera_matrix=K, dist_coeffs=D, image_size=np.array(image_size),
             rms=rms, mean_error=mean_err)

    print("\n===== RESULTS =====")
    print("Image size (w, h):", image_size)
    print("Camera matrix K:\n", np.round(K, 2))
    print("Distortion coefficients D:", np.round(D.ravel(), 5))
    print(f"RMS reprojection error : {rms:.3f} pixels")
    print(f"Mean per-image error   : {mean_err:.3f} pixels")
    worst = int(np.argmax(per_image))
    print(f"Worst image            : {used[worst]} ({per_image[worst]:.3f} px)")
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
