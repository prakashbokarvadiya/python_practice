    # Fraud Detection Project - Complete Example with Fixed Target Name and Error Handling

# 1. Import Libraries
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier
    from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    print("Libraries imported successfully.")
except ImportError as e:
    print(f"Error importing libraries: {e}")


# 2. Load Data
try:
    df = pd.read_csv(r'E:\tops\python\Fraud.csv')  # Replace with your actual data path
    print("Data loaded successfully.")
    print(df.head())
    print(df.info())
    print(df.describe())
except FileNotFoundError:
    print("Data file not found. Please check the file path.")
except Exception as e:
    print(f"Error loading data: {e}")


# 3. Missing Values Handling
try:
    print("Missing values in each column:\n", df.isnull().sum())

    # Fill missing numerical values with median
    df = df.fillna(df.median(numeric_only=True))
    print("Missing values handled with median imputation.")
except Exception as e:
    print(f"Error handling missing values: {e}")


# 4. Outlier Detection & Removal for 'amount'
try:
    if 'amount' in df.columns:
        plt.figure(figsize=(8,5))
        sns.boxplot(x=df['amount'])
        plt.title('Boxplot of Amount - Outlier Detection')
        plt.show()

        Q1 = df['amount'].quantile(0.25)
        Q3 = df['amount'].quantile(0.75)
        IQR = Q3 - Q1
        before_rows = df.shape[0]
        df = df[(df['amount'] >= Q1 - 1.5*IQR) & (df['amount'] <= Q3 + 1.5*IQR)]
        after_rows = df.shape[0]
        print(f"Removed {before_rows - after_rows} outlier rows from 'amount' column.")
    else:
        print("'amount' column not found. Skipping outlier removal.")
except Exception as e:
    print(f"Error in outlier detection/treatment: {e}")


# 5. Multicollinearity Check (VIF)
try:
    target_col = 'isFraud'  # Updated target column name

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    numeric_df = df.select_dtypes(include=[np.number])
    X_vif = numeric_df.drop(target_col, axis=1)

    vif_data = pd.DataFrame()
    vif_data['feature'] = X_vif.columns
    vif_data['VIF'] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
    print("Variance Inflation Factor (VIF) for each feature:")
    print(vif_data)

    # Optional: drop features with VIF > 10
    high_vif_features = vif_data[vif_data['VIF'] > 10]['feature'].tolist()
    if high_vif_features:
        print(f"Dropping high VIF features: {high_vif_features}")
        df = df.drop(columns=high_vif_features)
except Exception as e:
    print(f"Error in multicollinearity check: {e}")


# 6. Feature Selection & Train-Test Split
try:
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Drop high cardinality columns to avoid explosion (nameOrig, nameDest)
    high_cardinality_cols = ['nameOrig', 'nameDest']
    X = X.drop(columns=high_cardinality_cols, errors='ignore')

    # One-hot encode only categorical columns with low cardinality ('type')
    categorical_cols = ['type']
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Now split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale numeric features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Data split and scaling completed.")
except Exception as e:
    print(f"Error in feature selection/data split: {e}")



# 7. Model Training (Random Forest)
try:
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    print("Random Forest model training completed.")
except Exception as e:
    print(f"Error in model training: {e}")


# 8. Model Evaluation
try:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"ROC AUC Score: {roc_auc:.4f}")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()
except Exception as e:
    print(f"Error during model evaluation: {e}")


# 9. Feature Importance
try:
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_names = X.columns
        sorted_indices = np.argsort(importances)[::-1]

        print("Feature Importances:")
        for idx in sorted_indices:
            print(f"{feature_names[idx]}: {importances[idx]:.4f}")
    else:
        print("This model does not support feature importances.")
except Exception as e:
    print(f"Error extracting feature importance: {e}")


# 10. Interpretation Placeholder
"""
In this section (Markdown Cell in Jupyter Notebook), you should explain:

- Which features are most important for detecting fraud
- How the model performs on your test data (accuracy, precision, recall, AUC)
- Any limitations or biases observed
- Suggestions for model improvement and deployment
"""
