Plan: Measure It, segmentation and metric measurement

Project: Measure It
Company: xis.ai
Role: AI / Computer Vision Intern
Approach: Smartphone object segmentation and metric measurement

1. Project overview

The goal is a computer vision system that detects a smartphone in an image, segments its boundaries, and estimates its physical width and height in millimetres using a reference object of known dimensions.

Camera calibration will correct lens distortion, and a trained segmentation model will identify the smartphone.

## Assessment Timing

- Assessment started: 17 November 2026, 7:00 PM (PKT).
- Planning and initial setup completed: 7:46 PM.
- Submission deadline: 18 November 2026, 3:00 AM (PKT).
- Remaining implementation time at planning stage: 7 hours 14 minutes.

2. Approach and technologies
Programming: Python
Image processing: OpenCV and NumPy
Camera calibration: OpenCV checkerboard calibration
Reference detection: OpenCV ArUco marker
Image annotation: CVAT or Label Studio
Segmentation architecture: U-Net
Deep learning framework: PyTorch
Development environment: VS Code and Google Colab
Version control: Git and GitHub

U-Net suits pixel-level segmentation and can be trained on a custom dataset. The ArUco marker serves as the reference because OpenCV can detect its corners.

3. Implementation plan and time allocation
Stage	                                        Planned time
Project setup and documentation              	 30 minutes
Camera calibration and undistortion	             60 minutes
Image collection and dataset preparation	     60 minutes
Image annotation	                             60 minutes
U-Net training and evaluation	                 90 minutes
Measurement pipeline development	             75 minutes
Accuracy and undistortion testing	             60 minutes
Final documentation, recording and submission	 45 minutes
Total	                                          8 hours
4. Implementation sequence
Create a public GitHub repository and the dev-test01 branch.
Commit this plan and the fixed dataset split file list before writing implementation code.
Capture at least 20 checkerboard images and calibrate the camera using OpenCV.
Capture at least 70 images of the chosen smartphone with the calibrated camera.
Annotate the smartphone boundaries and prepare training, validation and testing datasets.
Train and evaluate a U-Net segmentation model on the custom dataset.
Build an inference pipeline that undistorts an image, segments the smartphone, detects the reference marker and converts pixel dimensions into millimetres.
Compare predicted measurements against manual measurements taken with a ruler or calliper.
Compare measurement errors between original and undistorted images.
Document results, limitations and incomplete objectives.
5. Scope and limitations

The first priority is the four foundational requirements: camera calibration, a custom dataset, a trained segmentation model and an image-to-millimetres pipeline.

Advanced condition testing, extensive confidence-threshold experiments and further performance optimization come only after the foundational pipeline works.

The test dataset will stay isolated from training, validation and model selection.

All reported metrics and measurement results will come from actual experiments. Any incomplete requirement will be documented as incomplete.

6. Risks and decisions

The main risks are limited assessment time, dataset annotation effort and model training time.

The plan uses U-Net instead of the prohibited Ultralytics YOLO or Roboflow models. A manually measured ArUco marker will provide the scale.

The rejected alternative is a pretrained object detector used without training on the custom dataset, because the assessment requires training or fine-tuning a segmentation model on self-collected images.

If time runs short, the priority stays a reproducible, working foundational pipeline.