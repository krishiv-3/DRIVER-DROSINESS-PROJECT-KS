import cv2
import os
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from pygame import mixer


# ============================================================
# INITIALIZE ALARM
# ============================================================

mixer.init()
sound = mixer.Sound("alarm.wav")


# ============================================================
# LOAD FACE DETECTOR
# ============================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# ============================================================
# LOAD MODEL
# ============================================================

model = load_model(
    os.path.join("models", "model.h5")
)

print("Model loaded successfully")
print("Input:", model.input_shape)
print("Output:", model.output_shape)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()


# ============================================================
# SETTINGS
# ============================================================

font = cv2.FONT_HERSHEY_SIMPLEX

CLOSE_THRESHOLD = 0.60
OPEN_THRESHOLD = 0.60

# Eyes need to remain closed for this long
DROWSY_TIME = 1.5

eyes_closed_start = None
alarm_on = False


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read camera.")
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    height, width = frame.shape[:2]

    # --------------------------------------------------------
    # Grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # --------------------------------------------------------
    # Face detection
    # --------------------------------------------------------

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(120, 120)
    )


    # Default values
    eye_state = "No Face"
    confidence = 0.0

    close_probability = 0.0
    open_probability = 0.0


    # ========================================================
    # PROCESS FACE
    # ========================================================

    if len(faces) > 0:

        # Take the largest face
        face = max(
            faces,
            key=lambda rect: rect[2] * rect[3]
        )

        fx, fy, fw, fh = face


        # ----------------------------------------------------
        # Draw face
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (fx, fy),
            (fx + fw, fy + fh),
            (255, 0, 0),
            2
        )


        # ====================================================
        # FIXED EYE REGIONS
        # ====================================================

        # These coordinates are percentages of the face box.
        #
        # Left side of image
        left_x1 = int(fx + fw * 0.12)
        left_x2 = int(fx + fw * 0.48)

        # Right side of image
        right_x1 = int(fx + fw * 0.52)
        right_x2 = int(fx + fw * 0.88)

        # Vertical eye region
        eye_y1 = int(fy + fh * 0.20)
        eye_y2 = int(fy + fh * 0.48)


        # Make sure coordinates are valid
        left_x1 = max(0, left_x1)
        left_x2 = min(width, left_x2)

        right_x1 = max(0, right_x1)
        right_x2 = min(width, right_x2)

        eye_y1 = max(0, eye_y1)
        eye_y2 = min(height, eye_y2)


        # ====================================================
        # CREATE TWO EYE CROPS
        # ====================================================

        left_eye = frame[
            eye_y1:eye_y2,
            left_x1:left_x2
        ]

        right_eye = frame[
            eye_y1:eye_y2,
            right_x1:right_x2
        ]


        # Draw eye regions
        cv2.rectangle(
            frame,
            (left_x1, eye_y1),
            (left_x2, eye_y2),
            (0, 255, 0),
            2
        )

        cv2.rectangle(
            frame,
            (right_x1, eye_y1),
            (right_x2, eye_y2),
            (0, 255, 0),
            2
        )


        # ====================================================
        # PREDICT EACH EYE
        # ====================================================

        predictions = []


        for eye in [left_eye, right_eye]:

            if eye.size == 0:
                continue


            # Convert to grayscale
            eye_gray = cv2.cvtColor(
                eye,
                cv2.COLOR_BGR2GRAY
            )


            # Resize to model input
            eye_gray = cv2.resize(
                eye_gray,
                (24, 24)
            )


            # Normalize
            eye_gray = (
                eye_gray.astype("float32")
                / 255.0
            )


            # (24,24)
            #     ↓
            # (24,24,1)
            eye_gray = np.expand_dims(
                eye_gray,
                axis=-1
            )


            # (24,24,1)
            #     ↓
            # (1,24,24,1)
            eye_gray = np.expand_dims(
                eye_gray,
                axis=0
            )


            # CNN prediction
            prediction = model.predict(
                eye_gray,
                verbose=0
            )[0]


            predictions.append(prediction)


        # ====================================================
        # COMBINE LEFT + RIGHT EYE
        # ====================================================

        if len(predictions) > 0:

            predictions = np.array(predictions)

            close_probability = float(
                np.mean(predictions[:, 0])
            )

            open_probability = float(
                np.mean(predictions[:, 1])
            )


            # ------------------------------------------------
            # Determine eye state
            # ------------------------------------------------

            if (
                close_probability > open_probability
                and close_probability >= CLOSE_THRESHOLD
            ):

                eye_state = "Closed"
                confidence = close_probability


            elif (
                open_probability > close_probability
                and open_probability >= OPEN_THRESHOLD
            ):

                eye_state = "Open"
                confidence = open_probability


            else:

                eye_state = "Uncertain"
                confidence = max(
                    close_probability,
                    open_probability
                )


    # ========================================================
    # DROWSINESS TIMER
    # ========================================================

    current_time = time.time()


    if eye_state == "Closed":

        if eyes_closed_start is None:

            eyes_closed_start = current_time


        closed_duration = (
            current_time -
            eyes_closed_start
        )


    else:

        eyes_closed_start = None
        closed_duration = 0.0


        # Stop alarm
        if alarm_on:

            try:
                sound.stop()
            except:
                pass

            alarm_on = False


    # ========================================================
    # ALARM
    # ========================================================

    if closed_duration >= DROWSY_TIME:

        if not alarm_on:

            try:
                sound.play(-1)
                alarm_on = True

            except Exception as e:

                print(
                    "Alarm error:",
                    e
                )


    # ========================================================
    # STATUS
    # ========================================================

    if closed_duration >= DROWSY_TIME:

        status = "DROWSY!"
        status_color = (0, 0, 255)

    elif eye_state == "Closed":

        status = "Eyes Closed"
        status_color = (0, 165, 255)

    elif eye_state == "Open":

        status = "Alert"
        status_color = (0, 255, 0)

    elif eye_state == "No Face":

        status = "No Face"
        status_color = (255, 255, 255)

    else:

        status = "Monitoring"
        status_color = (255, 255, 255)


    # ========================================================
    # INFORMATION PANEL
    # ========================================================

    cv2.rectangle(
        frame,
        (0, 0),
        (390, 190),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        frame,
        f"Eye: {eye_state}",
        (10, 30),
        font,
        0.7,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Close: {close_probability * 100:.1f}%",
        (10, 60),
        font,
        0.6,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Open: {open_probability * 100:.1f}%",
        (10, 88),
        font,
        0.6,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Confidence: {confidence * 100:.1f}%",
        (10, 116),
        font,
        0.6,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Closed: {closed_duration:.1f}s",
        (10, 144),
        font,
        0.6,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Status: {status}",
        (10, 174),
        font,
        0.65,
        status_color,
        2
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "Driver Drowsiness Detection",
        frame
    )


    # Q = quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

try:
    sound.stop()
    mixer.quit()
except:
    pass