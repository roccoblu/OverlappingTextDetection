import tensorflow as tf
import tensorflow.keras.utils as utils
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def create_dataset(X, y_index, displacement, visualize=False):

    X_dimensions = {
        "center": ((2,2), (2,2)),
        "horizontal": ((2,2), (0,4)),
        "vertical": ((0,4), (2,2)),
        "diagonal": ((0,4), (0,4))
    }

    overlap_dimensions = {
        "center" : ((2,2), (2,2)),
        "horizontal": ((2,2), (4,0)),
        "vertical": ((4,0), (2,2)),
        "diagonal": ((4,0), (4,0))
    }

    X_pad_dim = X_dimensions[displacement]
    overlap_pad_dim = overlap_dimensions[displacement]
    integers = list(range(10))

    final_data = []
    final_label = []
    '''
    horizontal:
  
    [1, 1]
    [1, 1]
  
    ==>
  
    [0, 0, 0, 0]
    [1, 1, 0, 0]
    [1, 1, 0, 0]
    [0, 0, 0, 0]
    '''

    for digit in tqdm(range(10)):
        other_digits = [x for x in integers if x != digit]
        rows = np.argwhere(y_index == digit)
        rows = rows[0:90]
        digit_label = utils.to_categorical(digit, 10)
        for i in range(90):
            image = X[rows[i]]
            image = np.squeeze(image)
            image = np.pad(image, X_pad_dim, 'constant', constant_values=0)
            for others in range(9):
                pick_other_digits = other_digits[others]
                pick_rows = np.argwhere(y_index == pick_other_digits)
                pick_rows = pick_rows[0:10]
                pick_label = utils.to_categorical(pick_other_digits, 10)
                for row in range(10):
                    overlapping_image = np.pad(np.squeeze(X[pick_rows[row]]), overlap_pad_dim, 'constant', constant_values=0)
                    final = np.maximum(image, overlapping_image)
                    label = digit_label + pick_label

                    if visualize:
                        fig = plt.figure()
                        subplot1 = fig.add_subplot(2, 2, 1)
                        subplot1.imshow(final.reshape(32, 32), cmap=plt.cm.gray_r)
                        return

                    final_data.append(final)
                    final_label.append(label)

        # Remove used rows from X, y_index to avoid reusing
        for idx in sorted(rows.flatten(), reverse=True):
            X = np.delete(X, idx, axis=0)
            y_index = np.delete(y_index, idx, axis=0)

    final_data = np.array(final_data)
    final_label = np.array(final_label)
    return final_data, final_label

if __name__ == "__main__":
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X = np.concatenate((X_train, X_test))
    y_index = np.concatenate((y_train, y_test))

    h_data, h_label = create_dataset(X.copy(), y_index.copy(), "horizontal")
    v_data, v_label = create_dataset(X.copy(), y_index.copy(), "vertical")
    d_data, d_label = create_dataset(X.copy(), y_index.copy(), "diagonal")
    c_data, c_label = create_dataset(X.copy(), y_index.copy(), "center")

    np.save("data/horizontal_data.npy", h_data)
    np.save("data/horizontal_label.npy", h_label)
    np.save("data/vertical_data.npy", v_data)
    np.save("data/vertical_label.npy", v_label)
    np.save("data/diagonal_data.npy", d_data)
    np.save("data/diagonal_label.npy", d_label)
    np.save("data/center_data.npy", c_data)
    np.save("data/center_label.npy", c_label)
