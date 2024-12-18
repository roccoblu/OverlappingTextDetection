import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils import load_all_data

# Defining the three CNN models for DIGITNET-rec ensemble architecture as described in the paper
def cnn_model_1():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(100, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def cnn_model_2():
    model = Sequential([
        Conv2D(64, (5, 5), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(100, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def cnn_model_3():
    model = Sequential([
        Conv2D(128, (7, 7), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(256, (3, 3), activation='relu'),
        Dropout(0.5),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(100, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":

    X, Y = load_all_data()
    X = X / 255.0 

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    model_1 = cnn_model_1()
    model_2 = cnn_model_2()
    model_3 = cnn_model_3()

    print("Training Model 1...")
    model_1.fit(X_train, y_train, epochs=20, batch_size=256, validation_split=0.1)
    
    print("Training Model 2...")
    model_2.fit(X_train, y_train, epochs=20, batch_size=256, validation_split=0.1)
    
    print("Training Model 3...")
    model_3.fit(X_train, y_train, epochs=20, batch_size=256, validation_split=0.1)

    y_pred_1 = model_1.predict(X_test)
    y_pred_2 = model_2.predict(X_test)
    y_pred_3 = model_3.predict(X_test)

    # Combining predictions using ensemble voting
    y_pred_ensemble = (y_pred_1 + y_pred_2 + y_pred_3) / 3
    y_pred_final = np.argmax(y_pred_ensemble, axis=1)

    y_test_labels = np.argmax(y_test, axis=1)

    ensemble_accuracy = accuracy_score(y_test_labels, y_pred_final)
    print(f"Ensemble Model Accuracy: {ensemble_accuracy * 100:.2f}%")
