"""Step 3: measure the distance from the camera to an ArUco marker.

Run:  python measure_distance.py --image images/test/marker_50cm.jpg --marker-cm 5 --true-cm 50
(--true-cm is optional: the distance you measured with a tape, to check accuracy.)
"""
import argparse
import os

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser(description="Distance to an ArUco marker")
    ap.add_argument("--image", required=True)
    ap.add_argument("--marker-cm", type=float, required=True, help="side of the BLACK marker square, in cm")
    ap.add_argument("--true-cm", type=float, default=None, help="real distance from tape measure (optional)")
    ap.add_argument("--calib", default="results/calibration.npz")
    ap.add_argument("--out", default="results/measured.png")
    args = ap.parse_args()

    data = np.load(args.calib)
    K, D = data["camera_matrix"], data["dist_coeffs"]
    calib_size = tuple(int(v) for v in data["image_size"])

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Cannot read {args.image}")
    h, w = img.shape[:2]
    if (w, h) != calib_size:
        raise SystemExit(f"Image is {(w, h)} but calibration used {calib_size}. Use the same camera/resolution.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        raise SystemExit("No ArUco marker found. Check light, focus, and that marker is fully visible.")

    # The marker's 4 corners in real life (cm), centre of marker = (0, 0, 0)
    s = args.marker_cm / 2.0
    obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float32)

    cv2.aruco.drawDetectedMarkers(img, corners, ids)
    for c, marker_id in zip(corners, ids.ravel()):
        ok, rvec, tvec = cv2.solvePnP(obj, c.reshape(4, 2), K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            continue
        distance = float(np.linalg.norm(tvec))  # straight-line distance, camera to marker centre
        msg = f"ID {marker_id}: distance = {distance:.1f} cm"
        if args.true_cm:
            err = abs(distance - args.true_cm) / args.true_cm * 100
            msg += f"  (true {args.true_cm:.1f} cm, error {err:.1f}%)"
        print(msg)
        cv2.drawFrameAxes(img, K, D, rvec, tvec, args.marker_cm * 0.75)
        pt = tuple(int(v) for v in c.reshape(4, 2)[0])
        cv2.putText(img, f"{distance:.1f} cm", (pt[0], max(pt[1] - 15, 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cv2.imwrite(args.out, img)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
