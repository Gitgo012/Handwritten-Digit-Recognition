# Machine Learning Notebooks Collection

This repository contains a collection of Jupyter notebooks demonstrating various machine learning techniques, including neural networks, decision trees, and ensemble methods. The notebooks are designed for educational purposes as part of the Coursera Machine Learning Course Labs.

## Project Overview

- **Handwritten Digit Recognition:** Neural network models for classifying handwritten digits from the MNIST dataset.
- **Decision Tree Ensemble:** Implementation and comparison of Decision Tree, Random Forest, and XGBoost classifiers on a heart failure prediction dataset.
- **Decision Tree (NumPy):** Manual implementation and visualization of decision tree splits using information gain on a toy dataset.
- **Decision Tree Revision:** Classification of mushrooms as edible or poisonous using decision trees and one-hot encoding on a toy dataset.

## Notebooks

### 1. Handwritten_digit_recognition.ipynb
- **Purpose:** Build, train, and evaluate neural networks (fully connected and CNN) for handwritten digit recognition using the MNIST dataset.
- **Techniques:** TensorFlow/Keras, NumPy, data visualization, model evaluation.

### 2. Decision_Tree_Ensemble.ipynb
- **Purpose:** Explore tree-based ensemble methods for classification.
- **Techniques:**
  - One-hot encoding with Pandas
  - Decision Tree, Random Forest, and XGBoost classifiers (scikit-learn, xgboost)
  - Heart failure prediction dataset from Kaggle
- **Dependencies:** pandas, scikit-learn, xgboost, kagglehub, matplotlib

### 3. Decision_Tree_numpy.ipynb
- **Purpose:** Visualize and understand how a decision tree splits data using information gain.
- **Techniques:**
  - Manual implementation of decision tree logic
  - Visualization with Matplotlib
  - Toy dataset (cat/not-cat classification)
- **Dependencies:** numpy, pandas, matplotlib

### 4. Decision_Tree_revision.ipynb
- **Purpose:** Classify mushrooms as edible or poisonous using decision trees.
- **Techniques:**
  - One-hot encoding and manual data manipulation
  - Toy mushroom dataset
  - Visualization with Matplotlib
- **Dependencies:** numpy, matplotlib

## Requirements

- Python 3.x
- Jupyter Notebook or JupyterLab
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- xgboost
- kagglehub
- Pillow
- OpenCV (optional, for image preprocessing)
- Seaborn (for confusion matrix visualization)

Install dependencies with:

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn xgboost kagglehub pillow opencv-python seaborn
```

## Usage

1. Open any of the notebooks (`.ipynb` files) in Jupyter Notebook or JupyterLab.
2. Run the cells sequentially to:
   - Load and visualize data
   - Build and train models
   - Evaluate and visualize predictions
   - (For Handwritten_digit_recognition.ipynb) Test the model on your own images (see code for details)

## Notes

- The notebooks demonstrate both high-level (Keras) and low-level (NumPy) implementations for educational purposes.
- Some notebooks use toy datasets for illustration; do not use them for real-world decision making.

## License

This project is for educational use as part of the Coursera Machine Learning Course Labs.
