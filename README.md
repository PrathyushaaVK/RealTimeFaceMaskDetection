# Face Mask Detection

This Face Mask Detection project uses a deep learning model based on MobileNetV2 to detect whether a person is wearing a face mask. It can be applied for real-time detection using a webcam or used with a dataset of images.

## Dependencies

Install the required packages before running the project:

```bash
pip install tensorflow keras imutils opencv-python numpy scikit-learn matplotlib
```
## Getting Started

### Dataset Preparation

1. **Data Collection**: Place the images in the `dataset` directory:
   - `dataset/with_mask`: Images of people wearing masks.
   - `dataset/without_mask`: Images of people without masks.
   
2. **Download Face Detector Model**:
   - Save `deploy.prototxt` and `res10_300x300_ssd_iter_140000.caffemodel` files in the `face_detector` folder. These files are available with OpenCV’s deep learning face detector.

### Training the Model

Run the following command to train the mask detector model:

```bash
python train_mask_detector.py
```

This script:
  - Loads and preprocesses the images.
  - Trains a MobileNetV2-based model on the prepared dataset.
  - Saves the trained model as mask_detector.model.
  - Generates a plot.png showing training/validation loss and accuracy.

### Real-time Mask Detection
To perform real-time mask detection using a webcam, run:

```bash
python detect_mask_video.py
```
This script:

  - Loads the trained mask detection model (mask_detector.model) and face detector.
  - Captures video from the webcam, processes each frame, and identifies whether detected faces have masks.
  - Displays bounding boxes and labels (mask/no mask) on the live video feed.
  - Press q to exit the video stream.

### Real-Time Detection in Video:
Ensure you have a webcam connected, or modify the video stream source to use a recorded video file.

### Results
After training, the model will save metrics like training and validation accuracy in plot.png. You can view this plot to analyze the model’s performance.
