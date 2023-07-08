# Text-Classification-on-Email
This project focuses on developing a model to classify emails based on their categories, including Science, Politics, Entertainment, and Crime. The implementation is carried out using Python and popular libraries such as Scikit-learn, Numpy, Pandas, Matplotlib, and NLTK (Natural Language Toolkit).

The main objectives of the project are as follows:

Dataset Preparation: The project starts with collecting and preprocessing a dataset of emails labeled with their respective categories. The preprocessing steps involve tokenization, removing stop words, and applying stemming or lemmatization techniques to reduce the dimensionality and improve classification accuracy.

Feature Extraction: To transform the textual data into numerical features, various methods are employed, such as bag-of-words representation, TF-IDF (Term Frequency-Inverse Document Frequency), or word embeddings like Word2Vec or GloVe. These techniques capture the semantic meaning and context of the text, enabling effective classification.

Model Development: Different classification algorithms are implemented to build the email classification model. The algorithms used in this project include decision tree, KNeighbors, XGBoost, and ensemble methods like Random Forest or Gradient Boosting. These algorithms are trained on the labeled email data to learn patterns and relationships between the text features and their corresponding categories.

Dimensionality Reduction: To handle high-dimensional data and potentially improve model performance, dimensionality reduction techniques like Principal Component Analysis (PCA) and Singular Value Decomposition (SVD) are applied. These techniques reduce the number of features while preserving important information, aiding in model training and inference.

Model Evaluation: The trained model is evaluated using various performance metrics such as accuracy. Cross-validation techniques are employed to ensure the model's generalizability and robustness. Additionally, the model's performance is assessed on a new email that is not part of the dataset to validate its effectiveness in real-world scenarios.

The project leverages data visualization techniques provided by Matplotlib to visualize the classification results, feature importance, or the distribution of emails across different categories.
