# Handwritten Digit Recognition

This project demonstrates handwritten digit recognition using the MNIST dataset and neural networks implemented in TensorFlow/Keras and NumPy. The notebook walks through building, training, and evaluating both fully connected and convolutional neural networks (CNNs) for digit classification.

## Project Structure

- **Handwritten_digit_recognition.ipynb**: Main Jupyter notebook containing all code, explanations, and results.

## Objectives

- Build a neural network to classify handwritten digits from the MNIST dataset.
- Implement both Keras-based and custom NumPy-based neural networks.
- Visualize predictions and analyze model performance.
- Experiment with a CNN for improved accuracy.

## Methods

- **Data**: Uses the MNIST dataset (28x28 grayscale images of digits 0-9).
- **Models**:
  - Fully connected neural network (Keras Sequential API)
  - Custom dense layers and forward propagation using NumPy
  - Convolutional Neural Network (CNN) using Keras
- **Evaluation**: Accuracy on training and test sets, confusion matrix, and visualization of predictions.

## Results

- **Fully Connected Model**: ~94% test accuracy
- **Custom NumPy Model**: ~94% test accuracy
- **CNN Model**: ~98% test accuracy

## Usage

1. Open `Handwritten_digit_recognition.ipynb` in Jupyter Notebook or JupyterLab.
2. Run the cells sequentially to:
   - Load and visualize the MNIST data
   - Build and train the models
   - Evaluate and visualize predictions
   - Test the model on your own images (see code for details)

## Requirements

- Python 3.x
- TensorFlow
- NumPy
- Matplotlib
- Pillow
- OpenCV (optional, for image preprocessing)
- Seaborn (for confusion matrix visualization)

Install dependencies with:

```bash
pip install tensorflow numpy matplotlib pillow opencv-python seaborn
```

## Notes

- You can test the trained models on your own 28x28 grayscale digit images by modifying the relevant cells in the notebook.
- The project demonstrates both high-level (Keras) and low-level (NumPy) neural network implementations for educational purposes.

## License

This project is for educational use as part of the Coursera Machine Learning Course Labs.
