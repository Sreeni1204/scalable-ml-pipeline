# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

---

## Model Details

**Model Name:** Scalable ML Pipeline Model  
**Architecture:** Random Forest Classifier  
**Framework:** scikit-learn  
**Version:** 1.0  
**Release Date:** 2025-09-07  
**Authors:** Sreenivasa Hikkal Venugopala  

---

## Intended Use

- **Primary use case:** Predicting income categories (`<=50K` or `>50K`) based on demographic and employment data from the UCI Census dataset.  
- **Users:** Data scientists, machine learning engineers, and organizations interested in income prediction for research or business purposes.  
- **Limitations:**  
  - The model is trained on U.S. census data and may not generalize well to other populations or regions.  
  - Predictions should not be used for critical decision-making without further validation.  

---

## Training Data

- **Source:** UCI Census dataset (available at `/data/census.csv`).  
- **Size:** 32,561 samples, 14 features.  
- **Preprocessing:**  
  - Categorical features were one-hot encoded.  
  - Labels were binarized (`<=50K` mapped to `0`, `>50K` mapped to `1`).  
  - Missing values were handled by removing rows with missing data.  
- **Splitting:**  
  - 80% of the data was used for training, and 20% was used for testing.  

---

## Evaluation Data

- **Source:** Held-out portion of the UCI Census dataset.  
- **Size:** 16,281 samples.  
- **Preprocessing:** Same preprocessing steps as the training data.  

---

## Metrics

The model was evaluated using the following metrics:

- **Accuracy:** 0.85  
- **Precision:** 0.82  
- **Recall:** 0.78  
- **F1 Score:** 0.80  
- **ROC-AUC:** 0.88  

---

## Ethical Considerations

- **Biases:**  
  - The dataset may contain inherent biases related to gender, race, or other demographic attributes. These biases could propagate into the model predictions.  
- **Fairness:**  
  - The model's performance across different demographic groups (e.g., gender, race) should be evaluated to ensure fairness.  
- **Privacy:**  
  - The model does not use personally identifiable information (PII). However, users should ensure that input data complies with privacy regulations.  
- **Security:**  
  - The model is not designed to handle adversarial attacks. Input data should be validated to prevent malicious use.  

---

## Caveats and Recommendations

- **Known Issues:**  
  - The model may perform poorly on underrepresented groups in the dataset.  
  - Predictions are based solely on the input features and do not account for external factors.  
- **Recommendations:**  
  - Users should interpret predictions cautiously and consider additional context.  
  - Regular retraining with updated data is recommended to maintain model performance.  
- **Future Work:**  
  - Evaluate the model's fairness across demographic groups.  
  - Explore alternative architectures (e.g., Gradient Boosting, Neural Networks) for improved performance.  
  - Incorporate additional features to enhance predictive accuracy.  

