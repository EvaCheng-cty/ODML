import cv2
import mediapipe as mp
import numpy as np
import pickle as pkl
from openai import OpenAI
import os

def get_words_from_char(charString):
  client = OpenAI()
  response = client.chat.completions.create(
      model="gpt-4o",
      messages=[
          {"role": "user", "content": f"You are a spelling correcting bot. Convert this string of characters into a sentence. some of the characters are wrong. Don't explain, just give me the answer. \n{charString}"},
          # {"role": "user", "content": charString}
      ]
  )
  return(response.choices[0].message.content)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, model_complexity=0)
mp_drawing = mp.solutions.drawing_utils

# Capture video from the webcam
cap = cv2.VideoCapture(0)

def parse_class(pred):
    if pred == 0:
        return "None"
    elif 1 <= pred <= 26:
        return chr(ord('A') + pred - 1)


# Reinitialize and load the model
# model = pkl.load(open("./rf_model.pkl", "rb"))
# model = pkl.load(open("./svc_model.pkl", "rb"))
model = pkl.load(open("./svc_model_moredata.pkl", "rb"))
# model = pkl.load(open("./rf_model_moredata.pkl", "rb"))

prev_letter = None
count = 0
global_counter = 0
checkingForRepeat = False
print("\n")

charString = ""
pred_char = "None"
score = 0
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty frame.")
        continue


    # Flip the image horizontally for a selfie-view display
    # image = cv2.flip(image, 1)

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if global_counter % 3 == 0:
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
                x_input = x_input.reshape(1, -1) / palm_length
                # x_input = x_input.reshape(1, -1)

                # make predictions
                pred_prob = model.predict_proba(x_input)[0]
                pred = np.argmax(pred_prob) + 1
                score = np.max(pred_prob)
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
                    checkingForRepeat = False
                    count = 0
                elif count == 10:
                    # at least 10 frames of the same letter
                    # print(pred_char)
                    charString += pred_char
                    checkingForRepeat = True
                elif count == 20:
                    if checkingForRepeat:
                        # repeat the letter
                        charString += pred_char
                        checkingForRepeat = False

                count += 1

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
    
    global_counter += 1

    font = cv2.FONT_HERSHEY_SIMPLEX
    # score = 0
    font_scale = 1
    color = (0, 255, 0)  # White color (B, G, R)
    thickness = 5
    cv2.putText(image, pred_char+" " +str(score), (image.shape[1] - 150, 30), font, font_scale, color, thickness)  # Top-right corner

    # put the charString on the screen
    font_scale = 2
    color = (0, 255, 0)
    thickness = 5
    cv2.putText(image, charString.replace(" ", "_"), (10, 30), font, font_scale, color, thickness)

    # Display the image
    cv2.imshow('MediaPipe Hands', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
