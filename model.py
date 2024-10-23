import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import streamlit as st

def load_data(filepath='preprocessed_data.pkl'):
    try:
        df = pd.read_pickle(filepath, compression='gzip')
        if df.empty:
            print("The dataset is empty after loading.")
            return pd.DataFrame()
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()


def preprocess_data(df):
    st.header("Data Preprocessing")
    st.write("### Initial Data Shape")
    st.write(df.shape)

    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    st.write("### Missing Values in Each Column")
    st.write(missing_values)

    cols_to_drop = missing_values[missing_values == df.shape[0]].index.tolist()
    if cols_to_drop:
        st.write(f"### Dropping Columns with All Missing Values: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    else:
        st.write("### No columns with all missing values to drop.")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    categorical_cols = df.select_dtypes(include=[object]).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    st.write(f"### Data Shape after Imputing Missing Values: {df.shape}")

    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna()
    st.write(f"### Data Shape after Dropping Remaining NaNs: {df.shape}")

    if df.shape[0] == 0:
        st.error("The dataset is empty after dropping NaNs.")
        return df

    if 'ScaleScore' not in df.columns:
        st.error("'ScaleScore' column not found in the dataset.")
        st.write(df.columns.tolist())
        return df

    return df

def correlation_analysis(df):
    st.header("Correlation Analysis")
    corr_matrix = df.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()

    st.write("### Correlation Matrix")
    st.image('correlation_matrix.png', use_column_width=True)

    corr_with_target = corr_matrix['ScaleScore'].abs().sort_values(ascending=False)
    st.write("### Top 10 Features Correlated with ScaleScore")
    st.write(corr_with_target.head(10))


def feature_importance(df, X_train, y_train):
    st.header("Feature Importance")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    joblib.dump(rf_model, 'model_deployment/student_performance_model.pkl')
    joblib.dump(X_train.columns.tolist(), 'model_deployment/model_features.pkl')

    importances = rf_model.feature_importances_
    feature_names = X_train.columns
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(20))
    plt.title('Top 20 Feature Importances')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

    st.write("### Feature Importance Plot")
    st.image('feature_importance.png', use_column_width=True)
    st.write("### Feature Importance Data")
    st.dataframe(feature_importance_df.head(20))


def model_evaluation(df, X_train, X_test, y_train, y_test):
    st.header("Model Evaluation")

    rf_model = joblib.load('model_deployment/student_performance_model.pkl')
    y_pred_rf = rf_model.predict(X_test)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2_rf = r2_score(y_test, y_pred_rf)
    st.write(f"**Random Forest RMSE:** {rmse_rf:.2f}")
    st.write(f"**Random Forest R² Score:** {r2_rf:.2f}")

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    r2_lr = r2_score(y_test, y_pred_lr)
    st.write(f"**Linear Regression RMSE:** {rmse_lr:.2f}")
    st.write(f"**Linear Regression R² Score:** {r2_lr:.2f}")

    gb_model = GradientBoostingRegressor(random_state=42)
    gb_model.fit(X_train, y_train)
    y_pred_gb = gb_model.predict(X_test)
    rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))
    r2_gb = r2_score(y_test, y_pred_gb)
    st.write(f"**Gradient Boosting RMSE:** {rmse_gb:.2f}")
    st.write(f"**Gradient Boosting R² Score:** {r2_gb:.2f}")

    models_evaluation = pd.DataFrame({
        'Model': ['Random Forest', 'Linear Regression', 'Gradient Boosting'],
        'RMSE': [rmse_rf, rmse_lr, rmse_gb],
        'R² Score': [r2_rf, r2_lr, r2_gb]
    })
    st.write("### Model Evaluation Summary")
    st.table(models_evaluation)

    return rf_model, y_pred_rf


def enhanced_clustering_analysis(df, X):
    st.header("Enhanced Clustering Analysis")
    kmeans = KMeans(n_clusters=5, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X)

    cluster_counts = df['Cluster'].value_counts().sort_index()
    st.write("### Cluster Counts")
    st.bar_chart(cluster_counts)

    st.write("### Cluster Summaries")
    for cluster in df['Cluster'].unique():
        cluster_df = df[df['Cluster'] == cluster]
        st.write(f"#### Cluster {cluster} Summary")
        st.write(cluster_df.describe())

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['Cluster'], palette='viridis', s=100)
    plt.title("PCA of Clusters")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.tight_layout()
    plt.savefig('clusters_pca.png')
    plt.close()

    st.image('clusters_pca.png', caption='PCA Visualization of Clusters', use_column_width=True)

    return df


def main():
    st.title("Student Performance Model Training and Analysis")

    df = load_data()

    if not df.empty:
        df = preprocess_data(df)

        if not df.empty:
            correlation_analysis(df)

            y = df['ScaleScore']
            X = df.drop(['ScaleScore'], axis=1)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            feature_importance(df, X_train, y_train)
            rf_model, y_pred_rf = model_evaluation(df, X_train, X_test, y_train, y_test)
            df = enhanced_clustering_analysis(df, X)


if __name__ == "__main__":
    main()
