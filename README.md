# Camera Calibration & Distance Measurement (Python, OpenCV)

Finds a camera's intrinsic parameters from chessboard photos, removes lens distortion,
and measures the real-world distance from the camera to an ArUco marker.

## What it does

1. **Calibrate** – detects chessboard corners in 15–20 photos and solves for the camera matrix `K`
   and distortion coefficients `D` (`cv2.calibrateCamera`), then reports the reprojection error.
2. **Undistort** – uses `K` and `D` to correct lens distortion and saves a before/after image.
3. **Measure distance** – detects an ArUco marker, estimates its 3D pose with `cv2.solvePnP`,
   and prints the camera-to-marker distance in cm.

## Results (my camera)

> Fill these in after running the project on your own photos.

| Item | Value |
|---|---|
| Camera / resolution | [e.g. phone model, 1920x1080] |
| Valid calibration images | [N] / [total] |
| RMS reprojection error | [X] px |
| Focal length (fx, fy) | [X], [X] px |
| Distance test | true [X] cm → measured [X] cm (error [X]%) |

![before and after](results/before_after.png)
![distance measurement](results/measured.png)

## Setup

```bash
pip install -r requirements.txt
python generate_patterns.py      # makes results/chessboard.png and results/marker_id0.png
```

Print both at 100% scale on flat paper, stick them on a flat surface, then measure with a ruler:
the size of one chessboard square and the black side of the marker.

## Usage

```bash
# 1. put 15-20 chessboard photos (different angles, same camera + resolution) in images/chessboard/
python calibrate.py --square-cm 2.5

# 2. undistort any photo taken with the same camera and resolution
python undistort.py --image images/test/photo.jpg

# 3. photo of the marker at a known distance (measured with a tape)
python measure_distance.py --image images/test/marker.jpg --marker-cm 5 --true-cm 50
```

## Project structure

```
generate_patterns.py   prints-ready chessboard + ArUco marker
calibrate.py           chessboard corners -> K, D, reprojection error
undistort.py           removes lens distortion, saves before/after
measure_distance.py    ArUco pose estimation -> distance in cm
```

## Notes and limitations

- Uses the standard pinhole + radial/tangential distortion model (suitable for phone and webcam lenses,
  not extreme fisheye).
- Use the same camera and the same resolution for calibration and measurement.
- Accuracy depends on printing the chessboard flat and measuring square/marker sizes correctly.
