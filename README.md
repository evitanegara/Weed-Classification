# Weed Classification Using Image Processing and Ensemble Learning
# Project Overview
This project focuses on classifying two types of weeds Charlock and Cleavers using image processing, machine learning, and ensemble methods. Weeds reduce crop yield by competing for essential resources and fostering pest infestations. Automating weed detection helps optimize herbicide use and improve agricultural efficiency. The project compares traditional classifiers (Logistic Regression, Random Forest, SVC) with ensemble approaches (Voting, Stacking) and a Convolutional Neural Network (CNN). Advanced feature extraction techniques (HOG, LBP, Color Histograms) are employed for traditional models, while CNN performs automatic feature learning.

## Dataset Overview
- Source: [Kaggle Plant Seedlings Classification](https://www.kaggle.com/competitions/plant-seedlings-classification/)
- Weed Types: Charlock and Cleavers
- Structure: Three folders train, validation, and test

| Subset     | Charlock | Cleavers | Total Images |
|------------|----------|----------|--------------|
| Train      | 272      | 208      | 480          |
| Validation | 90       | 58       | 148          |
| Test       | 90       | 68       | 158          |

## Project Workflow

### 1. Data Import and Setup
- Libraries used: **OpenCV**, **NumPy**, **Scikit-learn**, **Skimage**, **Matplotlib**, **TensorFlow**, **Keras**

### 2. Image Preprocessing and Feature Extraction
- Resized all images to 256×256 pixels
- Converted to grayscale
- Extracted features using:
  - HOG (Histogram of Oriented Gradients)
  - Color Histograms
  - Local Binary Patterns (LBP)

### 3. Feature Processing for Validation and Test Sets
- Applied the same preprocessing and feature extraction methods used for training data.

### 4. Model Development and Hyperparameter Tuning
- Trained the following models:
  - Logistic Regression (LR)
  - Random Forest (RF)
  - Support Vector Classifier (SVC)
  - Voting Classifier
  - Stacking Classifier
  - Convolutional Neural Network (CNN)
- Used GridSearchCV to optimize hyperparameters for LR, SVC, and RF.

### 5. Ensemble Learning
- Voting Classifier: Combined LR, SVC, and RF using majority voting
- Stacking Classifier: Used a meta-model to combine predictions from base learners

### 6. CNN Model
To further enhance classification accuracy, a Convolutional Neural Network (CNN) was implemented. CNNs are a powerful class of deep learning models specifically designed for image data. Architecture Details:
- 3 × Conv2D layers with filter sizes of 32, 64, and 128, using 3×3 kernels
- Each Conv2D layer is followed by a MaxPooling2D layer with a 2×2 pooling size
- ReLU activation was applied after each convolution layer to introduce non-linearity and reduce vanishing gradients
- The final Dense layer uses Softmax activation for binary classification (Charlock vs. Cleavers)
- This architecture allows the model to progressively learn low-to-high-level features, enhancing its ability to distinguish weed types based on shape, texture, and structure.
- 
## Model Evaluation Results
![image](https://github.com/user-attachments/assets/aa378c27-a24a-489b-be34-b8abf3ad0a51)
![image](https://github.com/user-attachments/assets/19569c02-7554-4268-ba03-e9f2c066ab3e)


## Confusion Matrix Visualization
Each confusion matrix below illustrates the classification breakdown for Charlock and Cleavers by model, offering a transparent view into how each algorithm handles false positives and false negatives.
![image](https://github.com/user-attachments/assets/7586cb9a-8bf8-49c4-a616-14ebf877fd9f)

## Highlights & Confusion Matrix Insights

- Voting Classifier achieved the highest accuracy (96%), correctly predicting 86 of 90 Charlock and 65 of 68 Cleavers, with only 7 total misclassifications, demonstrating exceptional consistency across both classes.

- Stacking Classifier followed closely with 95% accuracy, effectively integrating multiple base learners into a robust and generalizable meta-model.

- Support Vector Classifier (SVC) delivered a strong 94% accuracy, with particularly high recall for Charlock (96%), showcasing its strength in handling high-dimensional features.

- Logistic Regression reached 92% accuracy, showing that linear models, when combined with rich feature extraction like HOG and LBP — can be highly effective.

- Random Forest performed moderately at 89% accuracy, but misclassified 11 Charlock samples, indicating possible sensitivity to noise or overfitting on complex patterns.

- Convolutional Neural Network (CNN) outperformed all traditional models with approximately 97% accuracy. Its architecture — with 3 Conv2D layers, ReLU activation, and MaxPooling2D, enabled automatic extraction of hierarchical image features, leading to superior generalization.

## Key Takeaways

- Ensemble learning performs best :Voting and Stacking Classifiers surpassed individual models by leveraging diverse algorithm strengths, increasing accuracy and robustness.

- Deep feature engineering boosts classical models : Handcrafted features like HOG, LBP, and color histograms significantly improved the performance of models such as Logistic Regression and Random Forest.

- CNN delivers superior accuracy : With end-to-end feature learning from raw image data, CNNs eliminated the need for manual feature engineering while achieving the highest predictive performance.

- Interpretability vs. accuracy tradeoff : While models like Random Forest and Logistic Regression offer transparency, CNNs and ensembles prioritize accuracy with reduced interpretability.

## Contact
For any questions or inquiries, please contact evitanegara@gmail.com



