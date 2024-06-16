# Face Recognition System

![Face Recognition System](https://github.com/Jithendra1101/Face_recognization_system_RTP/images/bgimage1.png)

This project is a Python-based face recognition system using OpenCV and tkinter for graphical user interface (GUI).

## Overview

The face recognition system allows users to perform the following tasks:

- **Training**: Train the face recognition model using a dataset of images.
- **Detection**: Detect faces in real-time using a webcam or camera input.
- **Dataset Generation**: Generate a dataset of face images by capturing images from a webcam, associating them with user-provided IDs, names, ages, and addresses.

## Technologies Used

- **Python**: Main programming language.
- **OpenCV**: Library used for computer vision tasks, including face detection and recognition.
- **tkinter**: Python's de-facto standard GUI (Graphical User Interface) package.
- **PIL (Python Imaging Library)**: Library used for opening, manipulating, and saving image files.
- **NumPy**: Fundamental package for numerical computing in Python.

## Features

- **Face Detection**: Utilizes OpenCV's `CascadeClassifier` for frontal face detection.
- **LBPH Face Recognizer**: OpenCV's `LBPHFaceRecognizer` for training and recognizing faces.
- **GUI**: Developed using tkinter for an interactive user interface.
- **Data Persistence**: Stores user information (ID, name, age, address) in a text file (`user_info.txt`).

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/Jithendra1101/Face_recognization_system_RTP.git
   cd Face_recognization_system_RTP

