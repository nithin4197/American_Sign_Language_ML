import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model(r'ASL.h5')

# Initialize the video capture
video_capture = cv2.VideoCapture(0)  # You can specify the path to a video file instead of 0 for webcam

# Dictionary mapping class index to alphabet
class_to_alphabet = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'space',
    27: 'del', 28: 'nothing'
}


# Function to preprocess frames
def preprocess_frame(frame):
    # Resize frame to match model input size
    frame = cv2.resize(frame, (64, 64))
    # Convert to float32 and normalize
    frame = frame.astype('float32') / 255.0
    # Expand dimensions to match model input shape
    frame = np.expand_dims(frame, axis=0)
    return frame


while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if not ret:
        break

    # Preprocess the frame
    processed_frame = preprocess_frame(frame)

    # Perform prediction
    predictions = model.predict(processed_frame)
    # Get the predicted class label
    predicted_class = np.argmax(predictions)

    # Get the predicted alphabet
    predicted_alphabet = class_to_alphabet[predicted_class]

    # Display the predicted alphabet
    cv2.putText(frame, 'Predicted Alphabet: {}'.format(predicted_alphabet), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 255, 0), 2)

    # Display the frame with predicted class label
    cv2.imshow('Video', frame)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
video_capture.release()
cv2.destroyAllWindows()
