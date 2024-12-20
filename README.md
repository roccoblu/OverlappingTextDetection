# Overlapping Text Detection

This work is based on the paper by Huseyin Kusetogullari, Amir Yavariabdi, Johan Hall, Niklas Lavesson; [DIGITNET: A Deep Handwritten Digit Detection and Recognition Method Using a New Historical Handwritten Digit Dataset](http://dx.doi.org/10.1016/j.bdr.2020.100182).

## Abstract
Modern OCR systems struggle to accurately recognize overlapping text characters, particularly in historical or degraded documents. This project explores machine learning solutions to accurately detect and recognize overlapping digits by simplifying the problem to pairs of overlapped handwritten digits sourced from the MNIST dataset. Multiple methods are trained and compared: a traditional KNN classifier, CNN-based models (VGGNet, ResNet), and a more complex CNN+RNN architecture (DIGITNET). Results show that deep learning methods outperform the baseline KNN, consistently achieving high accuracy under simplified conditions. The experimental setup is constrained, but this work shows potential for more robust, real-world overlapping text recognition solutions.

## Introduction
Handwritten text recognition has come a long way, but overlapping characters remain a challenge. The problem of identifying two digits overlapped in a single image. Using the MNIST dataset is used to generate overlapping digit pairs in four different orientations (center, horizontal, vertical, diagonal). Multiple models are trained to find the most effective machine learning approach for this simplified problem.

## Methods
1. Data Generation:
Custom dataset is created by overlapping MNIST digits in different orientations. For each orientation, N examples of overlapped pairs are produced and store them as .npy files. Labels are represented as combined one-hot vectors.
2. Models:
- KNN: A traditional non-parametric baseline. Experiment with various distance metrics and neighbor counts.
- VGGNet (CNN): A standard deep CNN architecture (VGG16) adapted for grayscale images.
- ResNet (CNN): A residual network (ResNet50) to handle deeper representation and improved accuracy.
- DIGITNET (CNN+RNN): A more complex architecture using YOLO-based detection plus an ensemble CNN for recognition, aiming to separate and recognize overlapped digits.
3. Training and Evaluation: Each model is trained on the generated dataset and evaluated using accuracy metrics. Performance is compared across all four overlap orientations.

## Results
1. KNN: Achieves over 80% accuracy with optimal parameters but lags behind CNN-based methods.
2. VGGNet & ResNet: Both achieve ~98%+ accuracy after a few epochs, demonstrating that deep CNNs can handle the overlapped digits well under controlled conditions.
3. DIGITNET: Shows ~99% accuracy, performing very well on this simplified dataset.

| **Model**           | **Orientation/Metric** | **Accuracy**       | **Comments**                                                                 |
|----------------------|-------------------------|--------------------|-------------------------------------------------------------------------------|
| **KNN**             | Center Overwriting     | 82.5%             | Simple model with reasonable accuracy.                                       |
|                     | Diagonal Overwriting   | 85.6%             | Best performance among KNN tests.                                           |
|                     | Horizontal Overwriting | 84.3%             | Consistent performance.                                                     |
|                     | Vertical Overwriting   | 83.7%             | Slightly lower but still over 80%.                                          |
|                     | Minkowski Distance     | 85.6%             | Same as Euclidean, expected.                                                |
|                     | Euclidean Distance     | 85.6%             | Similar to Minkowski.                                                       |
|                     | Manhattan Distance     | 82.4%             | Slightly lower accuracy.                                                    |
|                     | Jaccard Distance       | 86.6%             | Highest accuracy for KNN.                                                   |
|                     | Chebyshev Distance     | 44.1%             | Performs poorly with this metric.                                           |
| **VGGNet**          | Center Overwriting     | 98.7%             | Achieved after 10 epochs, demonstrating CNN power.                          |
|                     | Other Orientations     | ~98.14%           | Consistently high accuracy for all orientations.                            |
| **ResNet50**        | Center Overwriting     | 97.37%            | High accuracy, leveraging deeper architecture.                              |
|                     | Diagonal Overwriting   | 99.59%            | Near-perfect recognition.                                                   |
|                     | Horizontal Overwriting | 99.81%            | Best performance among ResNet cases.                                        |
|                     | Vertical Overwriting   | 99.78%            | Consistently high performance across all orientations.                      |
| **DIGITNET (Model 1)** | All Overwriting Cases | 99.51%            | High accuracy, demonstrates the strength of the ensemble approach.          |
| **DIGITNET (Model 2)** | All Overwriting Cases | 98.81%            | Slightly lower than Model 1 but still very high.                            |
| **DIGITNET (Model 3)** | All Overwriting Cases | 99.49%            | Very high accuracy, consistent with Model 1.                                |
| **DIGITNET Ensemble** | Combined Prediction   | ~99.51% (average) | Ensemble of three CNNs ensures robustness and the highest overall accuracy. |
   
## Conclusion
All deep learning models significantly outperformed KNN on the simplified overlapping digit recognition task. However, due to the controlled nature of the data, these results may not be generalized to complex real-world overlapping texts. Future work is required to expanding beyond digits, introducing more complexity, and exploring additional architectures tailored to identifying multiple overlapped characters.

## Requirements
- Python 3.7+
- TensorFlow / Keras
- NumPy
- Scikit-learn
- Matplotlib

Install dependencies:
```
pip install -r requirements.txt
```

## Usage
1. Data Generation:
Run `data_generation.py` to regenerate the overlapping datasets if needed. This will produce .npy files in the `data/` directory.
```bash
python src/data_generation.py
```

2. Training Models:
- KNN:
```bash
python src/knn_experiment.py
```
- VGG:
```bash
python src/vgg_training.py
```
- ResNet:
```bash
python src/resnet_training.py
```
- DIGITNET:
```bash
python src/digitnet_training.py
```

Follow each script’s instructions/comments for parameters and paths.

3. YOLO Model for DIGITNET:
YOLO-related detection code is in yolo_model.py. yolov3.weights sohuld be placed in the appropriate directory as indicated in that file.

## Dataset
The `data/` directory contains .npy files for horizontal, vertical, diagonal, and center-overlapped datasets. Each .npy file can be loaded directly using `numpy.load()`.

## References
- Deep Residual Learning for Image Recognition (ResNet)
- CapsNet-Keras Github
- VGGNet Paper
- KNN Documentation by IBM
- YOLO: Real-Time Object Detection
