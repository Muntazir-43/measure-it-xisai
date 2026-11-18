
import cv2
import numpy as np
import glob
import os

# ---------------------------------------
# 1. Configuration
# ---------------------------------------

# Number of INNER corners (columns, rows)
CHECKERBOARD = (7, 9)

# Actual physical square size in millimetres
SQUARE_SIZE = 24.0

# Folder containing checkerboard images
IMAGE_FOLDER = "calibration/images"

# Output file for calibration parameters
OUTPUT_FILE = "calibration/camera_params.npz"

# ---------------------------------------
# 2. Prepare real-world checkerboard points
# ---------------------------------------

# Create a grid of 3D points.
# All checkerboard points lie on the same plane (Z = 0).

object_points_template = np.zeros(
    (CHECKERBOARD[0] * CHECKERBOARD[1], 3),
    np.float32
)

object_points_template[:, :2] = np.mgrid[
    0:CHECKERBOARD[0],
    0:CHECKERBOARD[1]
].T.reshape(-1, 2)

# Convert grid coordinates into millimetres
object_points_template *= SQUARE_SIZE

# Store 3D real-world points and corresponding 2D image points
object_points = []
image_points = []

# ---------------------------------------
# 3. Load checkerboard images
# ---------------------------------------

image_paths = sorted(
    glob.glob(os.path.join(IMAGE_FOLDER, "*.jpg"))
)

# Also support PNG images
image_paths += sorted(
    glob.glob(os.path.join(IMAGE_FOLDER, "*.png"))
)

print(f"Total images found: {len(image_paths)}")

if len(image_paths) < 20:
    raise ValueError(
        "At least 20 checkerboard images are required."
    )

image_size = None
successful_images = 0

# Sub-pixel corner refinement criteria
criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

# ---------------------------------------
# 4. Detect checkerboard corners
# ---------------------------------------

for image_path in image_paths:

    image = cv2.imread(image_path)

    if image is None:
        print(f"Could not read: {image_path}")
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    current_size = gray.shape[::-1]

    if image_size is None:
        image_size = current_size

    elif current_size != image_size:
        raise ValueError(
            "All calibration images must have the same resolution."
        )

    # Detect inner checkerboard corners
    found, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH +
        cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if found:

        # Refine corner positions to sub-pixel accuracy
        refined_corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria
        )

        object_points.append(
            object_points_template.copy()
        )

        image_points.append(refined_corners)

        successful_images += 1

        print(f"[OK] {os.path.basename(image_path)}")

    else:
        print(f"[FAILED] {os.path.basename(image_path)}")

print("\nSuccessfully detected checkerboards:",
      successful_images)

# Require all 20 or more images to be usable
if successful_images < 20:
    raise RuntimeError(
        "Fewer than 20 usable images. "
        "Capture additional checkerboard images."
    )

# ---------------------------------------
# 5. Perform camera calibration
# ---------------------------------------

print("\nCalibrating camera...")

rms, camera_matrix, distortion_coefficients, rvecs, tvecs = (
    cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None
    )
)


# ---------------------------------------
# 6. Calculate reprojection error per image
# ---------------------------------------

total_error = 0
per_image_errors = []

for i in range(len(object_points)):

    projected_points, _ = cv2.projectPoints(
        object_points[i],
        rvecs[i],
        tvecs[i],
        camera_matrix,
        distortion_coefficients
    )

    # Reshape both arrays and convert them to the same data type
    actual_points = image_points[i].reshape(-1, 2).astype(np.float64)

    projected_points = projected_points.reshape(-1, 2).astype(np.float64)

    # Calculate reprojection error
    error = np.linalg.norm(
        actual_points - projected_points,
        axis=1
    ).mean()

    total_error += error

    per_image_errors.append(
        (image_paths[i], error)
    )

mean_reprojection_error = (
    total_error / len(object_points)
)

print("\nReprojection error per image:")

for path, error in per_image_errors:
    print(f"{os.path.basename(path)}: {error:.4f} pixels")

print(
    f"\nMean Reprojection Error: "
    f"{mean_reprojection_error:.6f} pixels"
)

# ---------------------------------------
# 7. Display calibration results
# ---------------------------------------

print("\n========== CALIBRATION RESULTS ==========")

print("\nCamera Matrix:")
print(camera_matrix)

print("\nDistortion Coefficients:")
print(distortion_coefficients)

print(f"\nOpenCV Calibration RMS: {rms:.6f}")
print(
    f"Mean Reprojection Error: "
    f"{mean_reprojection_error:.6f} pixels"
)

print(f"\nImage Resolution: {image_size}")
print(f"Images Used: {successful_images}")

# ---------------------------------------
# 8. Save calibration parameters
# ---------------------------------------

os.makedirs("calibration", exist_ok=True)

np.savez(
    OUTPUT_FILE,
    camera_matrix=camera_matrix,
    distortion_coefficients=distortion_coefficients,
    image_width=image_size[0],
    image_height=image_size[1],
    checkerboard_columns=CHECKERBOARD[0],
    checkerboard_rows=CHECKERBOARD[1],
    square_size_mm=SQUARE_SIZE,
    calibration_rms=rms,
    mean_reprojection_error=mean_reprojection_error
)

print(f"\nCalibration parameters saved to: {OUTPUT_FILE}")

# ---------------------------------------
# 9. Create an undistorted preview
# ---------------------------------------

sample_image = cv2.imread(image_paths[0])

undistorted_image = cv2.undistort(
    sample_image,
    camera_matrix,
    distortion_coefficients
)

cv2.imwrite(
    "calibration/undistorted_preview.jpg",
    undistorted_image
)

print("Undistorted preview saved successfully.")
print("\nCamera calibration completed.")
