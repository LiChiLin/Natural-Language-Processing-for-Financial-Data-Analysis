# Advanced Machine Learning Applications for Finance

**Author:** Chi-Lin Li

## Project Overview

This project explores the use of advanced machine learning and natural language processing (NLP) techniques to enhance predictive performance in financial data analysis. Specifically, the project leverages pre-trained models, such as Word2Vec, Sentence BERT, and RoBERTa, along with various feature engineering methods, to extract meaningful insights from financial texts and improve model accuracy.

### Key Features:
- **Text-based Feature Engineering**: Utilizing Word2Vec, Sentence BERT, and RoBERTa for text embedding and feature extraction.
- **Optimization**: Implementing optimization techniques such as TF-IDF vectorization, cosine similarity, and model tuning to enhance predictive outcomes.
- **Machine Learning Models**: Comparing the performance of Random Forest, LightGBM, and XGBoost before and after model optimization.

## Methodologies

### 1. Preprocessing
- Handled financial fund data with gaps in certain features.
- Text cleaning, tokenization, and stopword removal to prepare for NLP tasks.
- Splitting the data into training (70%), validation (20%), and testing (10%) sets with reproducible results using `random_state=42`.

### 2. Skip-Gram Model & Word2Vec
- Trained a Word2Vec model with embedding dimensionality of 50, context window size of 3, and rare word exclusion for vocabulary enhancement.
- Developed a custom Word2Vec class to calculate cosine similarity between key financial terms and generate a knowledge base.

### 3. Classification Algorithms
- Implemented and compared machine learning models:
  - **Random Forest**: Improved with ensemble learning and bagging.
  - **LightGBM**: Used gradient-based one-sided sampling for memory efficiency.
  - **XGBoost**: Applied regularization to prevent overfitting.

### 4. NLP Models: Sentence BERT & RoBERTa
- Applied advanced transformers to extract rich semantic features from text data.
- Compared the performance of these models on financial texts in classification tasks.

## Results

### Pre-trained vs Self-trained Models
- **Before Optimization**: XGBoost performed the best with a precision of 0.65, recall of 0.68, and F1-score of 0.66.
- **After Optimization**: Random Forest achieved the best performance with a precision of 0.76 and an F1-score of 0.75.
- **Pre-trained Models**: RoBERTa consistently outperformed Sentence BERT, achieving the highest precision (0.80), recall (0.81), and F1-score (0.80) across all models.

### Knowledge Base & Semantic Analysis
- Built a directed graph using NetworkX to visualize relationships between financial terms based on cosine similarity.
- TF-IDF vectorization and cosine similarity measurements were used to analyze the relationships between financial summaries and the knowledge base.

## Conclusion

This project demonstrates the importance of embedding techniques, feature engineering, and optimization in improving the performance of machine learning models for financial data. The results emphasize that pre-trained models like RoBERTa provide enhanced predictive performance compared to self-trained models.

## Installation

To run this project, you will need to install the following dependencies:

```bash
pip install numpy pandas scikit-learn lightgbm xgboost transformers networkx
```

## Usage

1. **Preprocessing**: Prepare the financial dataset and perform data cleaning and tokenization.
2. **Train Word Embeddings**: Use the provided scripts to train the Word2Vec model and create embeddings.
3. **Build Knowledge Base**: Utilize the `CustomWord2Vec` class to calculate semantic similarities between key financial terms.
4. **Model Training**: Apply classification algorithms (Random Forest, LightGBM, XGBoost) and evaluate the performance.
5. **Evaluate Results**: Compare model performance before and after optimization, and analyze the output from pre-trained models.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contribution

Chi-Lin Li contributed 100% to this project.

---

You can customize this README further by adding specific file paths, detailed usage instructions, or code snippets relevant to your project.
