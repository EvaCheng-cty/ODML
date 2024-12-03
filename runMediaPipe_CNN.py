import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from openai import OpenAI

import json


# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Capture video from the webcam
cap = cv2.VideoCapture(0)


class CNNModel(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super(CNNModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, stride=1, padding=1),  # Output: 16 x 5 x 5
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),  # Output: 32 x 5 x 5
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 5 * 5, hidden_dim),  # Flattened input size
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),  # Flattened input size
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.layers(x)

def parse_class(pred):
    if pred == 0:
        return "None"
    elif 1 <= pred <= 26:
        return chr(ord('A') + pred - 1)

def keypoints_to_image(keypoint):
    image = torch.zeros((5, 5, 2))

        # Define the mapping as a 5x5 grid
    mapping = [
            [4, 8, 12, 16, 20],
            [3, 7, 11, 15, 19],
            [2, 6, 10, 14, 18],
            [1, 5, 9, 13, 17],
            [-1, -1, 0, -1, -1]  # -1 for x (zero-padding entries), 0 for index 0
        ]

    for i in range(5):
            for j in range(5):
                index = mapping[i][j]
                if index != -1:  # If not a padded zero
                    image[i, j, :] = torch.tensor(keypoint[index])  # Copy keypoint data (x, y, z)
        
    return image.permute(2, 0, 1)



input_dim = 42
hidden_dim = 1024   
num_classes = 26

# Reinitialize and load the model
loaded_model = CNNModel(hidden_dim, num_classes)
loaded_model.load_state_dict(torch.load("cnn_new_dataset_76.pth"))
loaded_model.eval()


def get_words_from_char(charString):
    client = OpenAI()
    response = client.chat.completions.create(
      model="gpt-4o",
      messages=[
          {"role": "user", "content": f"You are a spelling correcting bot. Convert this string of characters into a sentence. some of the characters are wrong. Don't explain, just give me the answer. \n{charString}"},
          # {"role": "user", "content": charString}
      ]
    )
    return (response.choices[0].message.content)


# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils

# Capture video from the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Failed to open camera!")
else:
    print("Camera initialized successfully!")


prev_letter = None
count = 0
print("\n")

charString = ""
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty frame.")
        continue


    # Flip the image horizontally for a selfie-view display
    image = cv2.flip(image, 1)

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and detect hands
    results = hands.process(image_rgb)

    # Draw the hand annotations on the image
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            x_input = np.array([[h.x, h.y] for h in hand_landmarks.landmark])

            # center about 9
            x_input = x_input - x_input[9]

            palm_length = np.linalg.norm(x_input[9] - x_input[0])

            # flatten
            x_input = x_input / palm_length

            x_input = torch.tensor(keypoints_to_image(x_input)).float().unsqueeze(0)

            # make predictions
            pred_prob = loaded_model(x_input)[0]
            pred = torch.argmax(pred_prob) + 1
            score = torch.max(pred_prob)
            # pred = model.predict(x_input)[0]
            # print(pred)
            pred_char = parse_class(pred)

            if prev_letter is None:
                prev_letter = pred_char
            elif prev_letter != pred_char:
                prev_letter = pred_char
                count = 0
            elif count > 50:
                # over 90 frames ~ 3 seconds
                # print("__")
                charString += " "
                prev_letter = None
                count = 0
            elif count == 20:
                # at least 10 frames of the same letter
                # print(pred_char)
                charString += pred_char

            count += 1

            font = cv2.FONT_HERSHEY_SIMPLEX
            # score = 0
            font_scale = 1
            color = (0, 255, 0)  # White color (B, G, R)
            thickness = 5
            cv2.putText(image, pred_char+" " +str(score.item()), (image.shape[1] - 150, 30), font, font_scale, color, thickness)  # Top-right corner

            # put the charString on the screen
            font_scale = 2
            color = (0, 255, 0)
            thickness = 5
            cv2.putText(image, charString.replace(" ", "_"), (10, 30), font, font_scale, color, thickness)

    else:
        if charString != "":
            # no hands
            # send to openai
            correctedString = get_words_from_char(charString)
            print("printing the original string")
            print(charString)
            print("printing the corrected string")
            print(correctedString)
            charString = ""
            prev_letter = None
            count = 0

    # Display the image
    cv2.imshow('MediaPipe Hands', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

