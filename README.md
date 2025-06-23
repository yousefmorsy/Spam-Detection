# Spam Email Detection Project

## Overview
This project implements a machine learning-based spam email detection system using various classification algorithms. The goal is to classify emails as spam or legitimate (ham) using natural language processing techniques and multiple machine learning models.

## Dataset
- **Source**: [Spam Emails Dataset](https://www.kaggle.com/datasets/abdallahwagih/spam-emails)
- **Shape**: 5,572 emails × 2 features
- **Features**: Email text and corresponding labels (spam/ham)

## Technologies Used
- **Python**: Core programming language
- **Jupyter Notebook**: Development environment
- **Libraries**:
  - `pandas`: Data manipulation and analysis
  - `numpy`: Numerical computing
  - `scikit-learn`: Machine learning algorithms and tools
  - `nltk`: Natural language processing
  - `xgboost`: Gradient boosting framework

## Text Preprocessing
The following preprocessing steps were applied to clean and prepare the email text data:

```python
import string
punc = string.punctuation

from nltk.corpus import stopwords
stop = stopwords.words("english")

from nltk.stem import WordNetLemmatizer
ps = WordNetLemmatizer()

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
```

### Preprocessing Pipeline:
1. **Punctuation Removal**: Removed special characters and punctuation
2. **Stopword Removal**: Filtered out common English stopwords
3. **Lemmatization**: Reduced words to their root forms
4. **Vectorization**: Converted text to numerical features using CountVectorizer and TfidfVectorizer

## Machine Learning Models
Six different classification algorithms were implemented and evaluated:

1. **Gaussian Naive Bayes** (`GaussianNB`)
2. **Multinomial Naive Bayes** (`MultinomialNB`)
3. **Bernoulli Naive Bayes** (`BernoulliNB`)
4. **Logistic Regression**
5. **Random Forest Classifier**
6. **XGBoost**

## Model Performance Results

### Overall Performance Metrics
| Model | Accuracy | Precision | F1 Score |
|-------|----------|-----------|----------|
| GaussianNB | 0.8574 | 0.8545 | 0.8545 |
| MultinomialNB | 0.9767 | 0.9880 | 0.9880 |
| BernoulliNB | **0.9835** | **0.9989** | **0.9989** |
| LogisticRegression | 0.9777 | 0.9755 | 0.9876 |
| RandomForest | 0.9758 | 0.9754 | 0.9865 |
| XGBoost | 0.9748 | 0.9733 | 0.9860 |

### Best Performing Model
- **Best Accuracy**: BernoulliNB (98.35%)
- **Best Precision**: BernoulliNB (99.89%)
- **Best F1 Score**: BernoulliNB (99.89%)

## Overfitting Analysis

### Training vs Test Performance
| Model | Train Accuracy | Test Accuracy | Difference | Status |
|-------|----------------|---------------|------------|---------|
| GaussianNB | 0.8886 | 0.8574 | 0.0312 | Slight Overfitting |
| MultinomialNB | 0.9920 | 0.9767 | 0.0153 | Good Generalization |
| **BernoulliNB** | **0.9874** | **0.9835** | **0.0039** | **Good Generalization** |
| LogisticRegression | 0.9949 | 0.9777 | 0.0172 | Good Generalization |
| RandomForest | 0.9998 | 0.9758 | 0.0240 | Slight Overfitting |
| XGBoost | 0.9903 | 0.9748 | 0.0155 | Good Generalization |

### Key Insights:
- **BernoulliNB** shows the best generalization with minimal overfitting (0.39% difference)
- **RandomForest** shows signs of overfitting with perfect training accuracy (99.98%)
- Most models demonstrate good generalization capabilities

## Cross-Validation Analysis (5-Fold)

| Model | CV Mean ± Std | Test Accuracy | CV vs Test Diff |
|-------|---------------|---------------|-----------------|
| GaussianNB | 0.8741 ± 0.0087 | 0.8574 | 0.0166 |
| MultinomialNB | 0.9854 ± 0.0023 | 0.9767 | 0.0087 |
| **BernoulliNB** | **0.9820 ± 0.0046** | **0.9835** | **-0.0015** |
| LogisticRegression | 0.9789 ± 0.0038 | 0.9777 | 0.0012 |
| RandomForest | 0.9769 ± 0.0033 | 0.9758 | 0.0012 |
| XGBoost | 0.9774 ± 0.0061 | 0.9748 | 0.0027 |

### Cross-Validation Insights:
- **BernoulliNB** shows excellent consistency with low standard deviation (0.46%)
- Test performance slightly exceeds CV mean for BernoulliNB, indicating robust performance
- **MultinomialNB** has the lowest variance across folds (0.23% std)

## Conclusions

1. **BernoulliNB** emerges as the best overall model with:
   - Highest accuracy (98.35%)
   - Exceptional precision (99.89%)
   - Best generalization capabilities
   - Consistent cross-validation performance

2. **Naive Bayes algorithms** (Multinomial and Bernoulli) perform exceptionally well for text classification tasks

3. **Random Forest** shows overfitting tendencies despite good test performance

4. All models except GaussianNB achieve over 97% accuracy, demonstrating the effectiveness of the preprocessing pipeline

## Usage
1. Clone the repository
2. Install required dependencies: `pip install pandas numpy scikit-learn nltk xgboost`
3. Download the dataset from Kaggle
4. Run the Jupyter notebook to reproduce results
5. Use the trained BernoulliNB model for new email classification

## Future Improvements
- Implement deep learning models (LSTM, BERT)
- Experiment with different text preprocessing techniques
- Add more sophisticated feature engineering
- Implement ensemble methods combining top-performing models
- Deploy the model as a web service or API

## License
This project is available under the MIT License.

## Contact
For questions or contributions, please open an issue in the repository.
