# 🚗 Driver Drowsiness Detection Using Deep Learning

A real-time computer vision and deep learning based system that detects driver drowsiness by monitoring the driver's eye state through a webcam. The system classifies the eyes as **OPEN** or **CLOSED** and triggers an audible alarm when the eyes remain continuously closed for a predefined duration.

---

## 📌 Project Overview

Driver drowsiness is one of the major causes of road accidents. A driver who is becoming sleepy may have slower reaction times and reduced attention.

This project aims to provide an automatic warning system by continuously monitoring the driver's face and eyes using a webcam.

The system follows this pipeline:

```text
Webcam
   ↓
Video Frame
   ↓
Face Detection
   ↓
Eye Region Extraction
   ↓
Image Preprocessing
   ↓
CNN Model
   ↓
Open / Closed Prediction
   ↓
Probability & Confidence Check
   ↓
Continuous Eye Closure Detection
   ↓
Drowsiness Detection
   ↓
Alarm
```

The project combines **Computer Vision**, **Deep Learning**, and **Real-Time Decision Logic**.

---

# 🎯 Objectives

The main objectives of this project are:

* Detect the driver's face in real time.
* Extract the driver's eye regions.
* Classify each eye as **Open** or **Closed**.
* Calculate the confidence/probability of the prediction.
* Monitor the duration of continuous eye closure.
* Detect possible driver drowsiness.
* Trigger an audible warning alarm.
* Provide a visual interface showing the current detection status.

---

# 🧠 Technologies Used

| Technology       | Purpose                                              |
| ---------------- | ---------------------------------------------------- |
| Python           | Main programming language                            |
| OpenCV           | Webcam handling, image processing and face detection |
| TensorFlow/Keras | Deep learning model and inference                    |
| NumPy            | Numerical and array operations                       |
| Pygame           | Alarm generation in `main.py`                        |
| Streamlit        | Browser-based user interface                         |
| Haar Cascade     | Face/eye detection                                   |
| MRL Eye Dataset  | Training data                                        |
| Keras `.h5`      | Saved trained model                                  |

---

# 📂 Project Structure

```text
Driver-Drowsiness-Detection-using-Deep-Learning/
│
├── main.py
├── streamlit_app.py
├── model_training.py
├── data_preparation.py
│
├── models/
│   ├── model.h5
│   └── model1.h5
│
├── haar_cascade_files/
│   └── Haar Cascade XML files
│
├── MRL Eye Data/
│   └── Dataset and prepared data
│
├── Data Preparation.ipynb
├── Model Training.ipynb
├── main.ipynb
│
├── alarm.wav
├── alarm1.wav
│
├── requirements.txt
├── packages.txt
├── Dockerfile
└── README.md
```

---

# 📁 File Description

## `main.py`

This is the **main real-time application**.

It:

1. Loads the trained CNN model.
2. Opens the webcam.
3. Detects the driver's face.
4. Extracts approximate left and right eye regions.
5. Preprocesses the eye images.
6. Sends the images to the CNN.
7. Gets Open/Closed probabilities.
8. Combines predictions from both eyes.
9. Applies a confidence threshold.
10. Measures continuous eye-closure duration.
11. Detects drowsiness.
12. Activates the alarm.
13. Displays information on the screen.

---

## `streamlit_app.py`

This file provides a **Streamlit-based web interface** for the project.

It performs the same general task as `main.py`, but presents the application through a browser-style interface.

The Streamlit version uses a frame-based closed-eye counter, while the main OpenCV application uses a time-based threshold.

---

## `data_preparation.py`

This script prepares the MRL Eye dataset.

The dataset filenames contain information about the eye state.

The project uses:

```python
i.split('_')[4]
```

to access the eye-state field.

The project interprets:

```text
0 → Closed Eye
1 → Open Eye
```

The images are then copied into their corresponding class directories.

---

## `model_training.py`

This file contains the model-training workflow.

It uses TensorFlow/Keras and includes:

* Image loading
* Image augmentation
* Training/validation split
* Model creation
* Loss function
* Optimizer
* Training
* Model checkpointing
* Early stopping
* Learning-rate reduction

The supplied training script contains an **InceptionV3 transfer-learning implementation**.

---

## `models/model.h5`

This is the trained model used by the current runtime application.

The actual saved model has a compact CNN architecture with an input shape of:

```text
24 × 24 × 1
```

where:

* 24 = image height
* 24 = image width
* 1 = grayscale channel

The final layer contains two Softmax outputs representing:

```text
Closed
Open
```

---

# 📊 Dataset

The project uses the **MRL Eye Dataset**.

The dataset contains eye images representing different eye states and conditions.

The project uses the images for binary classification:

```text
Open Eye
    vs
Closed Eye
```

The filename contains metadata that allows the eye-state label to be identified.

The prepared data is divided into appropriate class directories before training.

---

# 🧹 Data Preprocessing

Before an eye image is given to the CNN, several preprocessing operations are performed.

The pipeline is:

```text
Original Eye Crop
       ↓
Grayscale Conversion
       ↓
Resize to 24 × 24
       ↓
Normalize Pixel Values
       ↓
Add Channel Dimension
       ↓
Add Batch Dimension
       ↓
CNN Input
```

### 1. Grayscale Conversion

The eye image is converted from a color image to grayscale.

This reduces the image to one channel.

### 2. Resize

The image is resized to:

```text
24 × 24 pixels
```

because the trained runtime model expects this input size.

### 3. Normalization

Pixel values originally range approximately from:

```text
0 – 255
```

They are divided by 255:

```python
eye_gray = eye_gray.astype("float32") / 255.0
```

This produces values approximately between:

```text
0 – 1
```

### 4. Tensor Formatting

The image is converted to the required model input shape:

```text
1 × 24 × 24 × 1
```

The first dimension represents the batch size.

---

# 🧠 CNN Model

The actual runtime model is a Convolutional Neural Network.

Its general structure is:

```text
Input: 24×24×1
       ↓
Conv2D – 32 filters
       ↓
MaxPooling
       ↓
Conv2D – 32 filters
       ↓
MaxPooling
       ↓
Conv2D – 64 filters
       ↓
MaxPooling
       ↓
Dropout
       ↓
Flatten
       ↓
Dense – 128 neurons
       ↓
Dropout
       ↓
Dense – 2 neurons
       ↓
Softmax
       ↓
Closed / Open
```

### Convolution Layers

Convolution layers learn visual features from the eye images.

These features may include:

* Eyelid boundaries
* Eye opening
* Edges
* Intensity patterns
* Shapes and textures

### ReLU

ReLU is used as the activation function in the convolutional and Dense layers.

```text
ReLU(x) = max(0,x)
```

It introduces non-linearity into the network.

### Dropout

Dropout helps reduce overfitting by randomly disabling some neurons during training.

### Flatten

Flatten converts the learned feature maps into a one-dimensional vector.

### Dense Layer

The Dense layer combines the extracted features to make the final classification.

### Softmax

The final Softmax layer produces probabilities for the two classes:

```text
Closed
Open
```

Example:

```text
[0.85, 0.15]
```

can be interpreted as:

```text
Closed = 85%
Open   = 15%
```

---

# 👁️ Face and Eye Detection

OpenCV's Haar Cascade is used for face detection.

The system first searches for a face.

If multiple faces are detected, the largest face is selected because the project assumes that the largest/closest face belongs to the driver.

In the current `main.py`, the eye regions are estimated using fixed percentages of the detected face rectangle.

The approximate regions are:

```text
             Detected Face
      ┌───────────────────────┐
      │    LEFT    RIGHT      │
      │     👁       👁       │
      │                       │
      │          Nose         │
      │                       │
      └───────────────────────┘
```

---

# 🔍 Eye Classification

Each eye is passed through the CNN separately.

For example:

```text
Left Eye
→ Closed = 80%
→ Open = 20%

Right Eye
→ Closed = 70%
→ Open = 30%
```

The system averages the two predictions:

```text
Average Closed = 75%
Average Open   = 25%
```

Therefore:

```text
Eye State = CLOSED
```

---

# 🎚️ Confidence Threshold

The main application uses a confidence threshold of:

```text
0.60 = 60%
```

A prediction is accepted only when:

1. Its probability is higher than the other class.
2. Its probability is at least 60%.

For example:

```text
Closed = 80%
Open   = 20%
```

Result:

```text
CLOSED
```

But:

```text
Closed = 52%
Open   = 48%
```

Result:

```text
UNCERTAIN
```

because neither prediction reaches the 60% threshold.

---

# ⏱️ Drowsiness Detection Logic

The system does not trigger the alarm immediately when the eyes are classified as closed.

This is important because normal blinking also produces closed-eye frames.

The main application uses:

```text
DROWSY_TIME = 1.5 seconds
```

The logic is:

```text
Eyes Closed
     ↓
Start Timer
     ↓
Continue Monitoring
     ↓
Eyes Open?
   ↙       ↘
 YES       NO
 ↓          ↓
Reset      Continue Timer
Timer         ↓
          ≥ 1.5 seconds
               ↓
            DROWSY
               ↓
             ALARM
```

If the eyes open before 1.5 seconds:

```text
Timer → Reset
Alarm → Not triggered
```

---

# 🔊 Alarm System

The desktop application uses Pygame's mixer.

The alarm is loaded using:

```python
mixer.init()
sound = mixer.Sound("alarm.wav")
```

When drowsiness is detected:

```python
sound.play(-1)
```

The `-1` makes the alarm repeat continuously.

When the driver becomes alert:

```python
sound.stop()
```

The application also maintains an `alarm_on` Boolean variable so that the alarm is not unnecessarily restarted on every frame.

---

# 🖥️ Real-Time Display

The application displays useful information on the video feed, including:

* Eye state
* Closed probability
* Open probability
* Confidence
* Closed-eye duration
* Overall status

Possible states include:

```text
Alert
Eyes Closed
DROWSY!
No Face
Monitoring
Uncertain
```

This makes the system easier to monitor and debug.

---

# 🌐 Streamlit Application

The project also contains:

```text
streamlit_app.py
```

This provides a browser-based interface.

The general flow is:

```text
Streamlit Interface
       ↓
Start Webcam
       ↓
Capture Frames
       ↓
Face Detection
       ↓
Eye Detection
       ↓
CNN Prediction
       ↓
Closed/Open
       ↓
Drowsiness Score
       ↓
Alarm
```

The Streamlit version uses a closed-frame counter:

```text
CLOSED_FRAME_LIMIT = 45
```

This is different from the 1.5-second wall-clock timer used in `main.py`.

---

# ⚠️ Important Model Architecture Note

There is an important difference between the supplied training script and the actual runtime model.

The supplied `model_training.py` contains an InceptionV3-based architecture with an input of:

```text
80 × 80 × 3
```

However, the actual:

```text
models/model.h5
```

used by the current runtime application is a Sequential CNN with:

```text
24 × 24 × 1
```

input.

Therefore, the current runtime should be explained using the actual `model.h5` architecture and the preprocessing implemented in `main.py`.

The project archive appears to contain different versions/stages of the project.

---

# ▶️ Installation

Clone or download the project and open a terminal in the project directory.

Create and activate a Python environment.

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Make sure the following are available:

```text
models/model.h5
alarm.wav
```

Also make sure that the webcam is connected and accessible.

---

# ▶️ Running the Desktop Application

Run:

```bash
python main.py
```

The application will:

1. Open the webcam.
2. Detect the face.
3. Locate the eye regions.
4. Classify the eyes.
5. Display the prediction.
6. Monitor eye closure.
7. Trigger the alarm if prolonged closure is detected.

Press:

```text
Q
```

to exit the application.

---

# ▶️ Running the Streamlit Application

Run:

```bash
streamlit run streamlit_app.py
```

Then open the Streamlit URL shown in the terminal.

Start the webcam through the application interface.

---

# 🧪 Example

Suppose the CNN predicts:

```text
Left Eye:
Closed = 0.82
Open   = 0.18

Right Eye:
Closed = 0.78
Open   = 0.22
```

Average:

```text
Closed = 0.80
Open   = 0.20
```

Since:

```text
0.80 > 0.20
0.80 > 0.60
```

the eye state becomes:

```text
CLOSED
```

If this continues for:

```text
≥ 1.5 seconds
```

the system reports:

```text
DROWSY!
```

and activates the alarm.

---

# ⚠️ Limitations

The current system has several limitations:

* Haar Cascade detection can be affected by lighting and head pose.
* Fixed eye-region extraction assumes a relatively frontal face.
* Glasses and reflections can affect eye classification.
* Poor camera quality can reduce performance.
* The CNN can produce incorrect predictions.
* The system uses eye closure as a proxy for drowsiness.
* The 1.5-second threshold is a project-defined parameter and is not a medical diagnostic threshold.
* The Streamlit alarm implementation uses Windows-specific `winsound`.
* The supplied training script and final runtime model have different architectures.

---

# 🚀 Future Improvements

Possible improvements include:

1. Use facial landmarks for more accurate eye localization.
2. Use a modern deep-learning face detector.
3. Train the exact model architecture used during deployment.
4. Use subject-independent train/test splitting.
5. Add precision, recall, F1-score and confusion-matrix evaluation.
6. Add temporal smoothing.
7. Combine eye closure with blink rate.
8. Add head-pose detection.
9. Add configurable drowsiness thresholds.
10. Improve alarm and notification mechanisms.
11. Optimize the system for embedded devices.
12. Add event logging and performance monitoring.

---

# 📈 Evaluation

The model should ideally be evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

For a safety-oriented application, recall is particularly important because a false negative could mean that the system fails to warn a genuinely drowsy driver.

---

# 🔐 Safety Note

This project is an educational/prototype system and should not be considered a certified automotive safety system.

Real-world deployment would require extensive testing, validation, reliability analysis and safety certification.

---

# 🎓 Viva Summary

The easiest way to explain the complete project is:

```text
The webcam captures the driver.
        ↓
OpenCV detects the face.
        ↓
The eye regions are extracted.
        ↓
The eye images are converted to grayscale.
        ↓
They are resized to 24×24 and normalized.
        ↓
The CNN classifies them as Open or Closed.
        ↓
The predictions from both eyes are combined.
        ↓
A 60% confidence threshold is applied.
        ↓
Continuous eye closure is measured.
        ↓
If the eyes remain closed for 1.5 seconds,
the driver is considered drowsy.
        ↓
An alarm is activated.
```

---

# 📌 Key Values to Remember

| Parameter            | Value              |
| -------------------- | ------------------ |
| Dataset              | MRL Eye Dataset    |
| Classification       | Open vs Closed     |
| Runtime Model        | Sequential CNN     |
| Input Size           | 24 × 24            |
| Channels             | 1 / Grayscale      |
| Output Classes       | 2                  |
| Activation           | Softmax            |
| Confidence Threshold | 0.60               |
| Drowsiness Duration  | 1.5 seconds        |
| Main Application     | `main.py`          |
| Web Application      | `streamlit_app.py` |
| Runtime Model        | `models/model.h5`  |
| Alarm                | `alarm.wav`        |

---

# 👨‍💻 Author

**Driver Drowsiness Detection Using Deep Learning**

Developed as a computer vision and deep learning project demonstrating real-time eye-state classification and drowsiness alerting.
