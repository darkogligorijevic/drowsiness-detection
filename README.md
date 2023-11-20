# Drowsiness Detection Alert System

## Description
A Python-based computer vision project that detects drowsiness using facial landmarks.

## Overview
This project uses OpenCV, dlib, and other libraries to create a real-time drowsiness detection alert system. It analyzes facial landmarks to determine if the eyes are closed. If the eyes are closed, the user will be alerted.

## Features
- Real-time facial landmark detection.
- Eyes closed detection using the Eye Aspect Ratio (EAR) formula.
- Drowsiness alert with visual and auditory feedback.

## Getting Started

### Prerequisites

- Python 3.x
- Install dependencies with `pip install -r requirements.txt`

### Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/drowsiness-alert.git
    cd drowsiness-detection-alert
    ```

2. Run the main script:
    ```bash
    python main.py
    ```

3. Feel free to test the drowsiness detection alert system.

## Dependencies
- OpenCV
- dlib
- scipy
- pygame

### Install dependencies using:
```bash
pip install -r requirements.txt
```

## Acknowledgments
- The facial landmarks predictor model is provided by dlib.

## Configuration
The drowsiness alert system provides several configuration options that users can adjust to customize the behavior of the alert system. These options include:
- `EAR_THRESHOLD`: The threshold for detecting closed eyes.
- `ALERT_DURATION`: The duration (in seconds) for triggering the drowsiness detection alert.

To configure these options, users can modify the corresponding constants in the `main.py` script.

## Troubleshooting
If you encounter issues while running the drowsiness deetection alert system, consider the following troubleshooting steps:

1. **Issue: Eyes not detected**
   - **Solution:** Ensure proper lighting conditions and adjust the camera angle for better detection.

2. **Issue: False positives**
   - **Solution:** Tweak the `EAR_THRESHOLD` constant in the `main.py` script to fine-tune the sensitivity.
  
## Contributing 
Feel free to contribute to this project.

## License
This project is licensed under the [MIT License](https://github.com/darkogligorijevic/drowsiness-detection/blob/master/LICENSE).






