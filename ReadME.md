# Word For Thought

**Author:** Archit Agarwal   

This repository contains a collection of **machine learning projects** developed as part of the word for thought winter project organized by Brain and Cognitive Society under the SnT council. The projects cover **classification, regression, and signal processing**, with an emphasis on **model formulation, theoretical understanding, and end-to-end implementation**. 

The final project focuses on speech emotion recognition, where multiple model architectures are explored to address the task of identifying human emotional states from audio recordings. Speech emotion recognition is a challenging problem due to variability in speakers, speech content, recording conditions, and the subjective nature of emotions.

In this project, different feature representations and classification models are evaluated to understand how emotional information is encoded in speech signals. The work emphasizes the role of signal processing techniques and model selection in improving recognition performance.

Speech emotion recognition has important real-world applications in human–computer interaction, affective computing, mental health assessment, call center analytics, and assistive technologies, where understanding emotional cues can significantly enhance system responsiveness and user experience.

---

## Introduction

Machine learning provides a unified framework for modeling relationships between data and outcomes, spanning tasks such as classification, regression, and pattern recognition. The projects in this repository explore these tasks using classical machine learning models as well as feature-based pipelines.

Each project follows a structured approach:
- Problem formulation
- Mathematical modeling
- Model training and evaluation
- Practical implementation in Python

---

## Project Overview

This repository includes the following projects:

1. Animal Classification  
2. House Price Regression  
3. Linear Regression (from scratch)  
4. Logistic Regression for Classification  
5. Speech Emotion Recognition  

Each project is implemented as a standalone Jupyter notebook.

---

## 1. Animal Classification

### Description

This project addresses the problem of **animal image classification**, where the objective is to assign a semantic label to an input image based on visual content.

---

### Model and Theory

The task is formulated as a **multi-class supervised classification problem**:

\[
f : \mathbb{R}^{H \times W \times C} \rightarrow \{1, \dots, K\}
\]

where \(H, W, C\) denote image dimensions and \(K\) is the number of animal classes.

The model learns discriminative visual patterns through feature extraction and supervised learning, optimized using a classification loss such as cross-entropy.

---

### Pipeline Architecture

1. Image loading and preprocessing  
2. Normalization and feature preparation  
3. Train-test split  
4. Model training  
5. Performance evaluation  

---

### Results

- **Training accuracy:** ~91%  
- **Test accuracy:** **~88%**

These results demonstrate effective generalization on unseen animal images.

---

### Implementation

    jupyter notebook Animal_Classification_Project.ipynb

---

## 2. House Price Regression

### Description

This project implements a **regression-based model** to predict house prices from structured tabular data.

---

### Model and Theory

House price prediction is formulated as a supervised regression problem:

\[
y = f(x_1, x_2, \dots, x_n)
\]

where the objective is to minimize the **Mean Squared Error (MSE)** between predicted and actual prices.

---

### Pipeline Architecture

1. Data loading and cleaning  
2. Feature preprocessing and scaling  
3. Train-test split  
4. Regression model training  
5. Evaluation using MSE and \(R^2\)

---

### Results

The model captures underlying trends in the housing dataset and achieves reasonable predictive performance as measured by regression metrics.

---

### Implementation

    jupyter notebook house_prediction_model.ipynb

---

## 3. Linear Regression from Scratch

### Description

This project implements **linear regression from first principles**, focusing on understanding the underlying mathematics and optimization process.

---

### Model and Theory

The hypothesis function is:

\[
\hat{y} = w^T x + b
\]

The objective function is the **Mean Squared Error**:

\[
J(w, b) = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2
\]

Optimization is performed using **gradient descent**.

---

### Pipeline Architecture

1. Data preprocessing  
2. Feature normalization  
3. Gradient descent optimization  
4. Convergence analysis  

---

### Results

The implementation converges correctly and reproduces expected linear regression behavior.

---

### Implementation

    jupyter notebook "Linear Regression Project.ipynb"

---

## 4. Logistic Regression for Classification

### Description

This project implements **logistic regression** for binary classification tasks.

---

### Model and Theory

The model estimates class probabilities using the sigmoid function:

\[
P(y=1|x) = \sigma(w^T x + b)
\]

Training minimizes **binary cross-entropy loss** using gradient-based optimization.

---

### Pipeline Architecture

1. Data preprocessing  
2. Feature scaling  
3. Model training  
4. Decision boundary analysis  
5. Evaluation  

---

### Results

- **Classification accuracy:** **~94%**

The model achieves strong performance while maintaining probabilistic interpretability.

---

### Implementation

    jupyter notebook "Logistic Regression Project.ipynb"

---

## 5. Speech Emotion Recognition

### Description

This project focuses on **speech emotion recognition**, where the objective is to classify human emotional states (such as neutral, happy, sad, angry, etc.) from audio recordings. The task lies at the intersection of **signal processing** and **machine learning**, requiring the extraction of meaningful patterns from speech signals that correlate with emotional expression.

Speech emotion recognition has important real-world applications in **human–computer interaction**, **call center analytics**, **mental health assessment**, **affective computing**, and **assistive technologies**, where understanding a speaker’s emotional state can significantly improve system responsiveness and user experience.

---

### Model and Theory

Raw speech waveforms are first converted into compact numerical representations using established **audio feature extraction techniques**. Direct learning from raw audio signals is challenging due to high dimensionality and variability, making feature-based representations essential for classical machine learning approaches.

In this project, features such as **Mel-Frequency Cepstral Coefficients (MFCCs)**, **chroma features**, and **spectral contrast** are extracted from short-time frames of speech signals:

- **MFCCs** capture the short-term spectral envelope of speech on a perceptual (Mel) scale and approximate characteristics of human auditory perception.
- **Chroma features** represent the distribution of spectral energy across pitch classes.
- **Spectral contrast** highlights differences between spectral peaks and valleys, providing additional discriminative information.

After feature extraction, each audio sample is represented as a fixed-length feature vector. The emotion recognition task is then formulated as a **supervised multi-class classification problem**:

\[
f: \mathbb{R}^d \rightarrow \{1, 2, \dots, K\}
\]

where \(d\) denotes the feature dimension and \(K\) the number of emotion categories. A supervised classifier is trained on labeled feature vectors to learn the mapping from acoustic features to emotional states. Model parameters are optimized by minimizing a classification loss on the training set and evaluated on unseen data.

---

### Pipeline Architecture

1. **Audio signal loading**  
   - Load speech recordings and ensure consistent sampling rates.
2. **Feature extraction**  
   - Segment audio into short frames and extract MFCCs, chroma, and spectral features.
3. **Feature normalization**  
   - Standardize features to improve classifier stability and convergence.
4. **Model training**  
   - Train a supervised classifier on the extracted features.
5. **Evaluation**  
   - Evaluate performance using classification accuracy on a held-out test set.

---

### Results

- **Test accuracy:** **~40%**

The obtained accuracy reflects the **inherent difficulty of speech emotion recognition**, particularly with feature-based classical machine learning methods. Variability in speakers, recording conditions, and emotional expression introduces significant challenges, highlighting the importance of robust feature representations and dataset quality.

---

### Implementation

The complete implementation, including audio preprocessing, feature extraction, and classifier training, is provided in the following notebook:



