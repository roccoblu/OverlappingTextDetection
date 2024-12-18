import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from utils import load_all_data

if __name__ == "__main__":
    X, Y = load_all_data()
    # Y is a sum of two one-hot vectors. Each label has two '1's.
    # For KNN, transform them into a tuple label or similar representation.
    # Now just find the two positions with '1':

    def label_to_tuple(lbl):
        ones = np.where(lbl == 1)[0]
        return tuple(ones)

    Y_tuples = np.array([label_to_tuple(y) for y in Y])

    X_flat = X.reshape((X.shape[0], -1))  # Flatten for KNN

    X_train, X_test, y_train, y_test = train_test_split(X_flat, Y_tuples, test_size=0.2, random_state=42)

    # KNN with default params (you can adjust n_neighbors and metric)
    knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski')
    knn.fit(X_train, y_train)

    accuracy = knn.score(X_test, y_test)
    print("KNN Accuracy:", accuracy)
