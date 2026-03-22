# Machine Learning Pipeline Libraries

## 1. Data Collection & Ingestion

| Library | Module / Class | Purpose | Use Case |
|---|---|---|---|
| **pandas** | `pd.read_*` | Load structured data from CSV, SQL, JSON, Excel | Reading a CSV of customer transactions into a DataFrame for analysis |
| **sqlalchemy** | `sqlalchemy.create_engine` | Connect to and extract from SQL databases | Pulling user records from a PostgreSQL database into a pipeline |
| **beautifulsoup4** | `bs4.BeautifulSoup` | Parse HTML to scrape raw unstructured data | Scraping product names and prices from an e-commerce website |

---

## 2. Data Cleaning & Preprocessing

| Library | Module / Class | Purpose | Use Case |
|---|---|---|---|
| **pandas** | `pandas.DataFrame` | Handle missing values, filter, and manipulate data | Dropping rows with null labels and filling gaps in sensor readings |
| **numpy** | `numpy.ndarray` | Perform fast mathematical operations on arrays | Computing the log-transform of a skewed numerical feature |
| **scikit-learn** | `sklearn.impute` | Fill missing values (e.g., `SimpleImputer`) | Replacing missing age values with the column median before training |
| **scikit-learn** | `sklearn.preprocessing` | Scale, standardize, and encode data | Normalizing pixel values to [0, 1] before feeding into a neural network |

---

## 3. Feature Engineering

| Library | Module / Class | Purpose | Use Case |
|---|---|---|---|
| **scikit-learn** | `sklearn.decomposition` | Dimensionality reduction (e.g., PCA) | Reducing 500 image features down to 50 principal components to speed up training |
| **scikit-learn** | `sklearn.feature_selection` | Select the most predictive features statistically | Removing low-variance or redundant columns from a high-dimensional dataset |
| **category_encoders** | `category_encoders` | Advanced categorical encoding (e.g., Target Encoding) | Encoding a high-cardinality "city" column using target mean encoding |
| **featuretools** | `featuretools.dfs` | Automated feature creation (Deep Feature Synthesis) | Auto-generating aggregate features like "total spend per customer" from relational tables |
| **umap-learn** | `umap.UMAP` | Advanced non-linear dimensionality reduction | Visualizing clusters of high-dimensional text embeddings in 2D space |

---

## 4. Model Selection & Training

| Library | Module / Class | Purpose | Use Case |
|---|---|---|---|
| **scikit-learn** | `sklearn.model_selection` | Train/test splits, cross-validation, and Grid Search | Using 5-fold cross-validation to get a reliable accuracy estimate on a small dataset |
| **scikit-learn** | `sklearn.ensemble` | Train classic ML models (e.g., Random Forest) | Training a Random Forest to classify whether a loan applicant will default |
| **xgboost** (or lightgbm) | `xgboost.XGBClassifier` | Train high-performance gradient boosted trees | Winning a Kaggle tabular competition using an XGBoost ensemble |
| **optuna** | `optuna.create_study` | Automated and highly efficient hyperparameter tuning | Searching for the optimal learning rate and tree depth for an XGBoost model |
| **pytorch** (or tensorflow) | `torch.nn` | Build and train complex deep neural networks | Training a CNN to classify handwritten digits from the MNIST dataset |

---

## 5. Evaluation & Interpretability

| Library | Module / Class | Purpose | Use Case |
|---|---|---|---|
| **scikit-learn** | `sklearn.metrics` | Calculate Accuracy, F1-score, RMSE, ROC-AUC | Comparing the F1-score of two classifiers on an imbalanced fraud detection dataset |
| **shap** | `shap.Explainer` | Model interpretability and feature importance scoring | Explaining why a model flagged a specific loan application as high-risk |
| **yellowbrick** | `yellowbrick.classifier` | Visual diagnostic tools for ML model evaluation | Plotting a learning curve to diagnose overfitting in a classifier |
| **matplotlib** (or seaborn) | `matplotlib.pyplot` | Custom plotting for evaluation (e.g., Confusion Matrix) | Rendering a confusion matrix heatmap to review misclassification patterns |