# ASL Alphabet Recognition Using TensorFlow and OpenCV

## Overview

This Python script implements a real-time American Sign Language (ASL) alphabet recognition system using TensorFlow and OpenCV. The system captures frames from a webcam feed, preprocesses them, passes them through a trained neural network model, and displays the predicted ASL alphabet on the video stream.

## Features

- **Real-time Recognition**: The system provides instant recognition of ASL alphabets from webcam input.
- **Easy Integration**: The script is simple to install and use, requiring only common Python libraries.

## Requirements

- Python 3.x
- OpenCV (`pip install opencv-python`)
- NumPy (`pip install numpy`)
- TensorFlow (`pip install tensorflow`)

## Installation

1. **Clone Repository**: Clone this repository or download the script (`asl_alphabet_recognition.py`) directly.

2. **Install Dependencies**: Use pip to install the required dependencies:

    ```
    pip install opencv-python numpy tensorflow
    ```

3. **Trained Model**: Ensure you have downloaded the h5 file.

## Usage

1. **Connect Webcam**: Ensure your webcam is connected to your computer.

2. **Run Script**: Execute the following command to run the script:

    ```
    python asl_alphabet_recognition.py
    ```

3. **Recognition**: The webcam feed will open, displaying the video stream with predicted ASL alphabets overlaid.

4. **Exit**: Press the 'q' key to exit the program.

## Customization

- **Model Replacement**: Replace `ASL.h5` with your trained model. Ensure the model can classify ASL alphabets and is saved in HDF5 format.
  
- **Class Labels**: Modify the `class_to_alphabet` dictionary to match class indices with corresponding ASL alphabets in your model.

- **Preprocessing**: Adjust the `preprocess_frame()` function to match the input preprocessing required by your model.

## Notes

- **Lighting Conditions**: Ensure optimal lighting conditions for accurate recognition.
- **Background**: Ensure that the video background is plain and cluttered to help the model function better.
- **Input Source**: The script assumes webcam input (`cv2.VideoCapture(0)`), but you can specify a video file path instead.
- **Educational Use**: This script is primarily for educational purposes and may require optimization for production use.

## Contributions

Contributions to enhance features, improve performance, or fix issues are welcome. Please fork the repository and submit pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
