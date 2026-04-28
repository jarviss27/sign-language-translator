import os
import cv2
import mediapipe as mp
import pandas as pd

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True)

data = []

dataset = "dataset"

for label in os.listdir(dataset):
    folder = os.path.join(dataset, label)

    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        img = cv2.imread(path)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]

            row = []

            for lm in hand.landmark:
                row.extend([lm.x, lm.y, lm.z])

            row.append(label)
            data.append(row)

df = pd.DataFrame(data)
df.to_csv("features.csv", index=False)

print("Features Extracted")