import cv2
import os
import time
import numpy as np
import streamlit as st
import winsound

from tensorflow.keras.models import load_model


# ============================================================
# HAAR CASCADE CLASSIFIERS
# ============================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_eye.xml"
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚗",
    layout="centered"
)


# ============================================================
# DROWSINESS DETECTION FUNCTION
# ============================================================

def drowsiness_detection(model_path, alarm_sound="alarm.wav"):

    # --------------------------------------------------------
    # Load trained model
    # --------------------------------------------------------

    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"Could not load the model: {e}")
        return

    # --------------------------------------------------------
    # Check alarm file
    # --------------------------------------------------------

    alarm_path = os.path.abspath(alarm_sound)

    if not os.path.exists(alarm_path):
        st.warning(
            f"Alarm file not found: {alarm_path}"
        )

    # --------------------------------------------------------
    # Open webcam
    # --------------------------------------------------------

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error(
            "❌ Could not access your webcam. "
            "Make sure your camera is connected and not being "
            "used by another application."
        )
        return

    # --------------------------------------------------------
    # Detection variables
    # --------------------------------------------------------

    closed_frames = 0
    total_frames = 0

    alarm_playing = False

    # Number of consecutive closed-eye frames required
    # before alarm starts.
    #
    # At approximately 15-30 FPS, 45 frames is around
    # 1.5-3 seconds.
    #
    CLOSED_FRAME_LIMIT = 45

    # --------------------------------------------------------
    # Streamlit placeholders
    # --------------------------------------------------------

    frame_placeholder = st.empty()
    status_placeholder = st.empty()
    score_placeholder = st.empty()

    st.info(
        "Camera is running. Keep your face clearly visible."
    )

    # --------------------------------------------------------
    # Main webcam loop
    # --------------------------------------------------------

    while True:

        # ----------------------------------------------------
        # Read webcam frame
        # ----------------------------------------------------

        ret, frame = cap.read()

        if not ret:
            st.error("❌ Could not read frame from webcam.")
            break

        total_frames += 1

        height, width = frame.shape[:2]

        # ----------------------------------------------------
        # Convert frame to grayscale
        # ----------------------------------------------------

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # ----------------------------------------------------
        # Detect faces
        # ----------------------------------------------------

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )

        eye_predictions = []

        # ----------------------------------------------------
        # Process detected faces
        # ----------------------------------------------------

        for (x, y, w, h) in faces:

            # Draw face rectangle
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                2
            )

            # ------------------------------------------------
            # Extract face region
            # ------------------------------------------------

            roi_gray = gray[
                y:y + h,
                x:x + w
            ]

            # ------------------------------------------------
            # Detect eyes INSIDE the face
            # ------------------------------------------------

            eyes = eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(20, 20)
            )

            # ------------------------------------------------
            # Process eyes
            # ------------------------------------------------

            for (ex, ey, ew, eh) in eyes:

                # --------------------------------------------
                # Extract actual eye image
                # --------------------------------------------

                eye = roi_gray[
                    ey:ey + eh,
                    ex:ex + ew
                ]

                # --------------------------------------------
                # Safety check
                # --------------------------------------------

                if eye.size == 0:
                    continue

                # --------------------------------------------
                # Resize to model input
                #
                # MODEL EXPECTS:
                # (24, 24, 1)
                # --------------------------------------------

                eye = cv2.resize(
                    eye,
                    (24, 24)
                )

                # --------------------------------------------
                # Normalize pixels
                # --------------------------------------------

                eye = eye.astype(
                    "float32"
                ) / 255.0

                # --------------------------------------------
                # Add batch dimension
                #
                # (24,24)
                #       ↓
                # (1,24,24)
                # --------------------------------------------

                eye = np.expand_dims(
                    eye,
                    axis=0
                )

                # --------------------------------------------
                # Add grayscale channel
                #
                # (1,24,24)
                #       ↓
                # (1,24,24,1)
                # --------------------------------------------

                eye = np.expand_dims(
                    eye,
                    axis=-1
                )

                # --------------------------------------------
                # Make prediction
                # --------------------------------------------

                prediction = model.predict(
                    eye,
                    verbose=0
                )

                # --------------------------------------------
                # Store prediction
                #
                # prediction[0][0] = Closed probability
                # prediction[0][1] = Open probability
                # --------------------------------------------

                closed_probability = float(
                    prediction[0][0]
                )

                open_probability = float(
                    prediction[0][1]
                )

                eye_predictions.append(
                    (
                        closed_probability,
                        open_probability
                    )
                )

                # --------------------------------------------
                # Draw eye rectangle
                # --------------------------------------------

                cv2.rectangle(
                    frame,
                    (x + ex, y + ey),
                    (
                        x + ex + ew,
                        y + ey + eh
                    ),
                    (0, 255, 0),
                    2
                )

        # ====================================================
        # DETERMINE EYE STATUS
        # ====================================================

        if len(eye_predictions) > 0:

            # ------------------------------------------------
            # Average predictions from detected eyes
            # ------------------------------------------------

            avg_closed = np.mean(
                [
                    p[0]
                    for p in eye_predictions
                ]
            )

            avg_open = np.mean(
                [
                    p[1]
                    for p in eye_predictions
                ]
            )

            # ------------------------------------------------
            # Determine whether eyes are closed
            # ------------------------------------------------

            if avg_closed > 0.70:

                eye_status = "CLOSED"

                # Increase consecutive closed frame counter
                closed_frames += 1

            elif avg_open > 0.70:

                eye_status = "OPEN"

                # Reset when eyes open
                closed_frames = 0

            else:

                eye_status = "UNCERTAIN"

        else:

            # No eyes detected
            eye_status = "NO EYES DETECTED"

        # ====================================================
        # CALCULATE DROWSINESS SCORE
        # ====================================================

        drowsiness_score = min(
            100,
            int(
                (closed_frames /
                 CLOSED_FRAME_LIMIT) * 100
            )
        )

        # ====================================================
        # ALARM LOGIC
        # ====================================================

        if closed_frames >= CLOSED_FRAME_LIMIT:

            # -----------------------------------------------
            # Driver considered drowsy
            # -----------------------------------------------

            cv2.putText(
                frame,
                "DROWSINESS ALERT!",
                (
                    max(20, width // 2 - 180),
                    50
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3,
                cv2.LINE_AA
            )

            # -----------------------------------------------
            # Start alarm ONLY ONCE
            # -----------------------------------------------

            if (
                not alarm_playing
                and os.path.exists(alarm_path)
            ):

                try:

                    winsound.PlaySound(
                        alarm_path,
                        winsound.SND_FILENAME |
                        winsound.SND_ASYNC |
                        winsound.SND_LOOP
                    )

                    alarm_playing = True

                except Exception as e:

                    print(
                        "Alarm error:",
                        e
                    )

        else:

            # -----------------------------------------------
            # Driver is not currently drowsy
            # -----------------------------------------------

            if alarm_playing:

                try:

                    winsound.PlaySound(
                        None,
                        winsound.SND_PURGE
                    )

                except Exception:
                    pass

                alarm_playing = False

        # ====================================================
        # DISPLAY STATUS ON VIDEO
        # ====================================================

        if eye_status == "CLOSED":

            status_text = "Eyes: CLOSED"

            status_color = (
                0,
                0,
                255
            )

        elif eye_status == "OPEN":

            status_text = "Eyes: OPEN"

            status_color = (
                0,
                255,
                0
            )

        else:

            status_text = (
                "Eyes: " +
                eye_status
            )

            status_color = (
                255,
                255,
                255
            )

        # ----------------------------------------------------
        # Eye status
        # ----------------------------------------------------

        cv2.putText(
            frame,
            status_text,
            (10, height - 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # Drowsiness score
        # ----------------------------------------------------

        cv2.putText(
            frame,
            "Drowsiness: "
            + str(drowsiness_score)
            + "%",
            (10, height - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # ====================================================
        # SHOW STATUS IN STREAMLIT
        # ====================================================

        if closed_frames >= CLOSED_FRAME_LIMIT:

            status_placeholder.error(
                "🚨 DROWSINESS DETECTED — WAKE UP!"
            )

        elif eye_status == "CLOSED":

            status_placeholder.warning(
                "⚠️ Eyes are closed"
            )

        elif eye_status == "OPEN":

            status_placeholder.success(
                "✅ Driver is alert"
            )

        else:

            status_placeholder.info(
                "👁️ Waiting for clear eye detection..."
            )

        score_placeholder.write(
            f"**Drowsiness Score: "
            f"{drowsiness_score}%**"
        )

        # ====================================================
        # CONVERT BGR → RGB
        # ====================================================

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # ====================================================
        # DISPLAY WEBCAM
        # ====================================================

        frame_placeholder.image(
            frame_rgb,
            channels="RGB",
            use_container_width=True
        )

        # ====================================================
        # SMALL DELAY
        # ====================================================

        time.sleep(0.01)

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()

    cv2.destroyAllWindows()

    # Stop alarm
    if alarm_playing:

        try:

            winsound.PlaySound(
                None,
                winsound.SND_PURGE
            )

        except Exception:
            pass


# ============================================================
# MAIN STREAMLIT APP
# ============================================================

def main():

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    st.title(
        "🚗 Driver Drowsiness Detection"
    )

    st.write(
        "Real-time driver drowsiness detection "
        "using Deep Learning and OpenCV."
    )

    # --------------------------------------------------------
    # Information
    # --------------------------------------------------------

    st.markdown(
        """
        ### How it works

        **Webcam → Face Detection → Eye Detection → 
        Deep Learning Model → Drowsiness Detection → Alarm**

        The model classifies the eyes as:

        - 🟢 **Open**
        - 🔴 **Closed**

        The alarm activates only when closed eyes are detected
        continuously for several frames.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # Model path
    # --------------------------------------------------------

    model_path = os.path.join(
        "models",
        "model.h5"
    )

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not os.path.exists(model_path):

        st.error(
            "❌ model.h5 was not found."
        )

        st.write(
            "Expected location:"
        )

        st.code(
            model_path
        )

        return

    # --------------------------------------------------------
    # Check alarm
    # --------------------------------------------------------

    alarm_path = "alarm.wav"

    if not os.path.exists(alarm_path):

        st.warning(
            "⚠️ alarm.wav was not found. "
            "The detector will still work, but no alarm "
            "sound will be played."
        )

    # --------------------------------------------------------
    # Start webcam
    # --------------------------------------------------------

    if st.button(
        "🎥 Start Webcam",
        type="primary"
    ):

        drowsiness_detection(
            model_path=model_path,
            alarm_sound=alarm_path
        )


# ============================================================
# RUN APP
# ============================================================

if __name__ == "__main__":
    main()
# import cv2
# import os
# from tensorflow.keras.models import load_model
# import numpy as np
# import streamlit as st



# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    
# def drowsiness_detection(model_path, alarm_sound='alarm.wav'):
    
#     model = load_model(os.path.join("models", "model.h5"))


#     lbl=['Close', 'Open']

#     path = os.getcwd()
#     cap = cv2.VideoCapture(0)
#     font = cv2.FONT_HERSHEY_COMPLEX_SMALL
#     score = 0

#     while(True):
#         ret, frame = cap.read()
#         height,width = frame.shape[:2]

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         faces = face_cascade.detectMultiScale(gray,minNeighbors = 3,scaleFactor = 1.1,minSize=(25,25))
#         eyes = eye_cascade.detectMultiScale(gray,minNeighbors = 1,scaleFactor = 1.1)

#         cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )

#         for (x,y,w,h) in faces:
#             cv2.rectangle(frame, (x,y) , (x+w,y+h) , (255,0,0) , 3 )

#         for (x,y,w,h) in eyes:

#             # eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
#             eye = cv2.resize(eye, (24, 24))
#             eye = eye.astype("float32") / 255.0
#             eye = np.expand_dims(eye, axis=0)
#             eye = np.expand_dims(eye, axis=-1)

#             prediction = model.predict(eye)
#             # print(prediction)
#         #Condition for Close
#             if prediction[0][0]>0.30:
#                 cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
#                 cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
#                 score=score+1
#                 #print("Close Eyes")
#                 if(score > 20):
#                     try:
#                         sound.play()
#                     except:  # isplaying = False
#                         pass

#             #Condition for Open
#             elif prediction[0][1] > 0.70:
#                 score = score - 1
#                 if (score < 0):
#                     score = 0
#                 cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
#                 #print("Open Eyes")
#                 cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)

#         cv2.imshow('frame',frame)
    
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
        
#     cap.release()
#     cv2.destroyAllWindows()


# # Streamlit app
# def main():
#     st.title("Emotion Recognition App")
    
#     # Click the button to start the webcam
#     if st.button("Start Webcam"):
#         drowsiness_detection(model_path = os.path.join('models', 'model.h5'))
        

# # Run the Streamlit app
# if __name__ == '__main__':
#     main()