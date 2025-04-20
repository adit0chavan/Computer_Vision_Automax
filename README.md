# Computer Vision Automax

A computer vision project for object detection using YOLOv8 and Roboflow for data labeling and model training.

## Project Overview

This project implements a computer vision system for automatic object detection in video feeds. It utilizes YOLOv8 for object detection and tracking, and Roboflow for dataset management and labeling.

## Team Members

- **Saumya Shah** - [LinkedIn](https://www.linkedin.com/in/saumya-shah-9b2579273/)
- **Bhavin Baldota** - [LinkedIn](https://www.linkedin.com/in/bhavin-baldota-103553234/)
- **Aditya Chavan** - [LinkedIn](https://www.linkedin.com/in/aditya-chavan-5117aa268/)

### Key Features

- Object detection and tracking in video streams
- Custom dataset creation and annotation using Roboflow
- Model training with YOLOv8
- Real-time object counting capabilities
- Video processing for frame extraction

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [Making Predictions](#making-predictions)
- [Results](#results)
- [Demo Video](#demo-video)
- [Team Members](#team-members)
- [License](#license)

## Project Structure

```
Computer_Vision_Automax/
├── Training/                      # Model training files
│   ├── train.ipynb                # Jupyter notebook for model training
│   ├── frame.py                   # Utility script for extracting frames from videos
│   └── runs/                      # Training results and model outputs
│       └── detect/
│           └── train/
│               ├── weights/       # Trained model weights
│               ├── Predict.py     # Script for making predictions
│               └── results.*      # Training metrics and visualizations
├── Test Video/                    # Test video files
│   └── testvid.mp4                # Sample test video with detection results
├── Input Videos/                  # Input video files for processing
│   └── input.mp4                  # Sample input video
├── Output.mp4                     # Output video with detections
├── README.md                      # Project documentation
└── LICENSE                        # License information
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Computer_Vision_Automax.git
cd Computer_Vision_Automax
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

*Note: The requirements.txt file should include ultralytics, roboflow, opencv-python, and other necessary packages.*

## Usage

### Data Preparation

1. Collect video footage for your use case
2. Extract frames from the videos using the frame extraction script:
```bash
python Training/frame.py
```
3. Create a Roboflow account and upload your frames
4. Label your data using Roboflow's annotation tools
5. Export your dataset in YOLOv8 format

### Model Training

1. Open the `Training/train.ipynb` notebook
2. Update the Roboflow API key and project details
3. Run the notebook to download your dataset and train the YOLOv8 model

The training process includes:
- Downloading the labeled dataset from Roboflow
- Setting up the YOLOv8 model
- Training for 50 epochs
- Validating model performance
- Saving the best weights

### Making Predictions

To run inference on a video:

```bash
cd Training/runs/detect/train/
python Predict.py
```

Update the video path in `Predict.py` to use your own video file.

## Results

The model achieves high performance metrics:
- mAP50: ~0.98
- Precision: ~0.96
- Recall: ~0.97

Training results, including charts and visualizations, are available in the `Training/runs/detect/train/` directory.

### Training Metrics Visualization

![Training Results](/Training/runs/detect/train/results.png)

The above chart shows the model's performance metrics throughout the training process, including box loss, classification loss, and mean Average Precision.

### Real-life Application Results

![Real-life Detection Results](/Training/runs/detect/train/output.png)

This image demonstrates the model's object detection capabilities in a real-world scenario, showing bounding boxes around detected objects along with confidence scores.

## Demo Video

The repository includes test and result videos demonstrating the model's performance:

- **Input Video**: Located in `Input Videos/input.mp4`
- **Test Video with Results**: Located in `Test Video/testvid.mp4` - This video shows the model's detection capabilities in action
- **Output Video**: `Output.mp4` contains the final processed video with object detection bounding boxes and counts

These videos serve as practical demonstrations of the model's object detection capabilities in real-world scenarios.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Roboflow](https://roboflow.com/)
- [OpenCV](https://opencv.org/) 
