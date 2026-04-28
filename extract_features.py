import os
import cv2
import mediapipe as mp
import pandas as pd

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True)

def process_dataset(base_folder, output_csv):
    data = []

    for label in os.listdir(base_folder):
        folder = os.path.join(base_folder, label)

        for file in os.listdir(folder):
            path = os.path.join(folder, file)

            img = cv2.imread(path)
            if img is None:
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]

                row = []
                for lm in hand.landmark:
                    row.extend([lm.x, lm.y, lm.z])

                row.append(label)
                data.append(row)

    pd.DataFrame(data).to_csv(output_csv, index=False)
    print(f"{output_csv} created.")

process_dataset("dataset/ISL", "features_isl.csv")
process_dataset("dataset/ASL", "features_asl.csv")