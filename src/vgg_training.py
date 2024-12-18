import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from utils import load_all_data

if __name__ == "__main__":
    X, Y = load_all_data()
    # Convert X to 3-channel by stacking (VGG expects 3 channels)
    X = np.stack((X,)*3, axis=-1)
    # Normalize images to [0,1]
    X = X / 255.0

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

    base_model = VGG16(weights=None, include_top=False, input_shape=(32,32,3))
    model = Sequential([
        base_model,
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')  # 10 outputs since we want the combined label vector
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_split=0.1, epochs=10, batch_size=256)
    loss, acc = model.evaluate(X_test, y_test)
    print("VGG Model Accuracy:", acc)
