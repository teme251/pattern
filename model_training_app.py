import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    IsolationForest,
    AdaBoostRegressor,
    BaggingRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import (
    KMeans,
    DBSCAN,
    AgglomerativeClustering,
    SpectralClustering,
    Birch,
    MeanShift,
    OPTICS,
)
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    explained_variance_score,
)
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress specific warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def load_data(filepath='preprocessed_data.pkl'):
    try:
        df = pd.read_pickle(filepath, compression='gzip')
        if df.empty:
            st.error("The dataset is empty after loading.")
            return pd.DataFrame()
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def preprocess_data(df):
    st.header("Data Preprocessing")
    st.write("### Initial Data Shape")
    st.write(df.shape)

    # Since preprocessing is already done, we might skip this step
    # But if you need any additional preprocessing, include it here

    if 'ScaleScore' not in df.columns:
        st.error("'ScaleScore' column not found in the dataset.")
        st.write(df.columns.tolist())
        return pd.DataFrame()

    return df

def correlation_analysis(df):
    st.header("Correlation Analysis")

    corr_matrix = df.corr()
    st.write("### Correlation Matrix")
    st.write(corr_matrix)

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()
    st.image('correlation_heatmap.png', caption='Correlation Heatmap', use_column_width=True)

    # Identifying highly correlated features
    threshold = 0.8
    corr_pairs = corr_matrix.unstack().sort_values(kind="quicksort")
    high_corr = corr_pairs[(abs(corr_pairs) > threshold) & (abs(corr_pairs) < 1)]
    st.write("### Highly Correlated Feature Pairs (Threshold > 0.8)")
    st.write(high_corr)

def feature_engineering(X, y):
    st.header("Feature Engineering")

    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # Feature Selection using SelectKBest
    selector = SelectKBest(score_func=f_regression, k='all')
    selector.fit(X_scaled, y)
    scores = pd.DataFrame({'Feature': X.columns, 'Score': selector.scores_})
    scores = scores.sort_values(by='Score', ascending=False)
    st.write("### Feature Scores using SelectKBest")
    st.write(scores)

    # Selecting top k features
    k = min(5, len(scores))  # Ensure k does not exceed the number of features
    top_features = scores.head(k)['Feature'].tolist()
    X_selected = X_scaled[top_features]

    st.write(f"### Shape of X_selected after Feature Selection: {X_selected.shape}")

    return X_selected

def model_evaluation(df, X_train, X_test, y_train, y_test):
    st.header("Model Evaluation")

    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
        'ElasticNet Regression': ElasticNet(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'Extra Trees': ExtraTreesRegressor(random_state=42),
        'AdaBoost': AdaBoostRegressor(random_state=42),
        'Bagging Regressor': BaggingRegressor(random_state=42),
        'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5),
        'Support Vector Regressor': SVR(kernel='linear'),
        'MLP Regressor': MLPRegressor(random_state=42, max_iter=1000, learning_rate_init=0.001),
    }

    evaluation_metrics = {
        'Model': [],
        'RMSE': [],
        'MAE': [],
        'R² Score': [],
        'Explained Variance': [],
        'Cross-Val Score': [],
    }

    for model_name, model in models.items():
        # Using Pipeline for scaling
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        evs = explained_variance_score(y_test, y_pred)

        # Cross-validation
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores.mean())

        evaluation_metrics['Model'].append(model_name)
        evaluation_metrics['RMSE'].append(rmse)
        evaluation_metrics['MAE'].append(mae)
        evaluation_metrics['R² Score'].append(r2)
        evaluation_metrics['Explained Variance'].append(evs)
        evaluation_metrics['Cross-Val Score'].append(cv_rmse)

    models_evaluation = pd.DataFrame(evaluation_metrics)
    st.write("### Model Evaluation Summary")
    st.table(models_evaluation)

    return models  # Return all trained models for further use

def hyperparameter_tuning(X_train, y_train):
    st.header("Hyperparameter Tuning")

    # Example for Random Forest Regressor
    param_grid = {
        'model__n_estimators': [50, 100, 150],
        'model__max_depth': [None, 10, 20],
        'model__min_samples_split': [2, 5, 10],
    }

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(random_state=42))
    ])

    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    st.write("### Best Hyperparameters (Random Forest):")
    st.write(grid_search.best_params_)

def enhanced_clustering_analysis(df, X):
    st.header("Enhanced Clustering Analysis")

    # Standardizing the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # List of clustering algorithms
    n_clusters = min(5, max(2, len(df) // 50))  # Adjust as appropriate
    clustering_algorithms = {
        'KMeans': KMeans(n_clusters=n_clusters, random_state=42),
        'AgglomerativeClustering': AgglomerativeClustering(n_clusters=n_clusters),
        'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
        'SpectralClustering': SpectralClustering(n_clusters=n_clusters, assign_labels='discretize', random_state=42),
        'GaussianMixture': GaussianMixture(n_components=n_clusters, random_state=42),
        'Birch': Birch(n_clusters=n_clusters, threshold=0.3),  # Adjusted threshold
        'MeanShift': MeanShift(),
        'OPTICS': OPTICS(min_samples=5, max_eps=2.0, cluster_method='xi'),  # Adjusted parameters
    }

    for name, algorithm in clustering_algorithms.items():
        try:
            df[f'{name}_Cluster'] = algorithm.fit_predict(X_scaled)
            cluster_counts = df[f'{name}_Cluster'].value_counts().sort_index()
            st.write(f"### {name} Cluster Counts")
            st.bar_chart(cluster_counts)
        except Exception as e:
            st.write(f"### {name} encountered an error: {e}")

    # Determine the number of PCA components
    n_components = min(2, X_scaled.shape[1])
    if n_components < 2:
        st.warning("Not enough features for PCA visualization.")
        return df

    # Visualize Clusters using PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # t-SNE visualization
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)

    # Plotting PCA clusters for KMeans
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['KMeans_Cluster'], palette='viridis', s=50)
    plt.title("PCA of Clusters (KMeans)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.tight_layout()
    plt.savefig('clusters_pca_kmeans.png')
    plt.close()
    st.image('clusters_pca_kmeans.png', caption='PCA Visualization of Clusters (KMeans)', use_column_width=True)

    # Plotting t-SNE clusters for KMeans
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=df['KMeans_Cluster'], palette='viridis', s=50)
    plt.title("t-SNE Visualization of Clusters (KMeans)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.tight_layout()
    plt.savefig('clusters_tsne_kmeans.png')
    plt.close()
    st.image('clusters_tsne_kmeans.png', caption='t-SNE Visualization of Clusters (KMeans)', use_column_width=True)

    return df

def anomaly_detection(df, X):
    st.header("Anomaly Detection")

    # Standardizing the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso_forest = IsolationForest(contamination=0.01, random_state=42)
    df['Anomaly'] = iso_forest.fit_predict(X_scaled)

    num_anomalies = (df['Anomaly'] == -1).sum()
    st.write(f"### Number of Anomalies Detected: {num_anomalies}")

    anomalies_df = df[df['Anomaly'] == -1]
    st.write("### Sample Anomalies")
    st.write(anomalies_df.head())

    # Visualizing anomalies with t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=df['Anomaly'], palette='coolwarm', s=50)
    plt.title('Anomaly Detection using t-SNE')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.tight_layout()
    plt.savefig('anomalies_tsne.png')
    plt.close()
    st.image('anomalies_tsne.png', caption='Anomaly Detection Visualization with t-SNE', use_column_width=True)

def main():
    st.title("Advanced Student Performance Model Training and Analysis")

    df = load_data()

    if not df.empty:
        df = preprocess_data(df)

        if not df.empty:
            correlation_analysis(df)

            y = df['ScaleScore']
            X = df.drop(['ScaleScore'], axis=1)

            X_selected = feature_engineering(X, y)

            if X_selected.shape[1] < 1:
                st.error("Not enough features selected after feature engineering.")
                return

            X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

            model_evaluation(df, X_train, X_test, y_train, y_test)
            hyperparameter_tuning(X_train, y_train)
            df = enhanced_clustering_analysis(df, X_selected)
            anomaly_detection(df, X_selected)

if __name__ == "__main__":
    main()
