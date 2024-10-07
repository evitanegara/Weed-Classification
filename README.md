# Business Understanding 
Weed detection is crucial for improving crop productivity and efficient agricultural management. Weeds compete with crops for essential resources like water and sunlight and contribute to the spread of diseases and pests. Automating weed classification through image processing represents a significant advancement in agriculture, helping farmers manage crops more effectively. 

# Projective Objective 
- The goal of this project is to classify two types of weeds—Charlock and Cleves—using image processing and machine learning techniques. The dataset is adapted from a Kaggle competition and consists of three folders: train, validation, and test. Check here for the dataset.
- Single classifiers like logistic regression and SVC were compared with ensemble learning techniques such as random forest, stacking, and voting to improve classification accuracy.
- By automating weed classification, this project aims to enhance decision-making in weed management, ultimately improving agricultural productivity and resource management.

# Project Overview 
This project focuses on classifying two types of weeds—Charlock and Cleves—using image processing and machine learning techniques. The key steps include:
- **Data Import**: Used libraries such as OpenCV, NumPy, scikit-learn, and skimage.
- **Pre- Processing and Extract Features for Train Data:** Images were resized, converted to grayscale, and features were extracted using HOG, color histograms, and LBP.
- **Pre- Processing and Extract Features for Validation and Test Set:**
The same preprocessing and feature extraction techniques were applied to the validation and test datasets.
- **Hyperparameter Tuning**: Conducted hyperparameter tuning for models such as logistic regression, SVC, and random forest to optimize performance for weed classification.
- **Building Models**:
  - Trained single classifiers including logistic regression and SVC, which were chosen for their ability to handle the complexity of the dataset. Random forest was employed to improve classification accuracy through its robust decision tree ensemble.
  - Applied stacking and voting ensemble methods to combine multiple classifiers, leveraging their strengths to enhance accuracy and model robustness. In addition, this project used CNN for weed classification. 
- **Evaluation** : The performance of the models was evaluated using a confusion matrix along with accuracy, precision, recall, and F1-score metrics.

# Project Result
-	Logistic Regression achieved 92% accuracy, Random Forest reached 91%, and SVC delivered 94%, with stable recall and precision across both weed types.
-	The Voting Classifier achieved the highest accuracy at 96%, followed closely by the Stacking Classifier with 95%, both showing balanced performance and strong generalization.
-	CNN: Outperformed traditional machine learning models with a 97% accuracy, attributed to its automatic feature extraction capabilities and ability to handle complex patterns in the data.
  
# Conclusion : 
The weed classification project successfully applied image processing and machine learning, with logistic regression, SVC, and random forest exceeding 90% accuracy. Ensemble learning, particularly the voting classifier, reached 96% accuracy, while CNN achieved 97% due to advanced feature extraction. Hyperparameter tuning and techniques like HOG, color histograms, and LBP significantly improved model accuracy.
