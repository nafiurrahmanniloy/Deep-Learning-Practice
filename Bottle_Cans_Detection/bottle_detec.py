import cv2
import numpy as np
import pyautogui
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load your saved model
model = load_model('bottle_can_classifier_final.h5')
class_labels = ['Defective', 'Proper', 'Can']

# Get full screen resolution
screen_width, screen_height = pyautogui.size()
print(f"[INFO] Capturing full screen: {screen_width}x{screen_height}")
print("[INFO] Starting detection. Press 'q' to quit.")

while True:
    # Take a full-screen screenshot
    screenshot = pyautogui.screenshot(region=(0, 0, screen_width, screen_height))
    frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    try:
        # Resize and preprocess image
        resized = cv2.resize(frame, (224, 224))
        image_array = img_to_array(resized) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Run prediction
        prediction = model.predict(image_array)[0]
        confidence = np.max(prediction)
        label = class_labels[np.argmax(prediction)]

        # Only show label if confidence is high enough
        if confidence > 0.8:
            color = (0, 255, 0) if label == 'Proper' else (0, 0, 255) if label == 'Defective' else (255, 165, 0)

            # Draw bounding box with label
            h, w, _ = frame.shape
            padding = 40
            startX, startY, endX, endY = padding, padding, w - padding, h - padding

            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 3)
            label_text = f"{label}: {confidence * 100:.2f}%"
            cv2.putText(frame, label_text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        # Display the frame
        cv2.imshow("Bottle & Can Detection", frame)

    except Exception as e:
        print("[ERROR]", e)
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Quitting.")
        break

cv2.destroyAllWindows()
