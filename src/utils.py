# code to load data generated by data_generation.py into data folder
import numpy as np

def load_data(orientation):
    # orientation: "horizontal", "vertical", "diagonal", "center"
    data = np.load(f"data/{orientation}_data.npy")
    labels = np.load(f"data/{orientation}_label.npy")
    return data, labels

def load_all_data():
    # Loads all orientations and concatenates them
    orientations = ["horizontal", "vertical", "diagonal", "center"]
    data_list = []
    label_list = []
    for o in orientations:
        d, l = load_data(o)
        data_list.append(d)
        label_list.append(l)
    return np.vstack(data_list), np.vstack(label_list)
