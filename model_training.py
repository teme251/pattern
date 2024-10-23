import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.ensemble import IsolationForest

def train_and_analyze():
    # Load preprocessed data
    df_model = pd.read_pickle('preprocessed_data.pkl', compression='gzip')
    print(f"Data Shape: {df_model.shape}")

    # Step 1: Inspect Missing Data
    missing_values = df_model.isnull().sum()
    print("\nMissing values in the dataset:")
    print(missing_values[missing_values > 0])

    # Step 2: Drop Columns with All Missing Values
    cols_to_drop = missing_values[missing_values == df_model.shape[0]].index
    print(f"\nDropping columns with all missing values: {cols_to_drop.tolist()}")
    df_model = df_model.drop(columns=cols_to_drop)

    # Step 3: Handle Remaining Missing Data (Imputation)
    # Impute missing numeric values with column mean
    numeric_cols = df_model.select_dtypes(include=[np.number]).columns
    df_model[numeric_cols] = df_model[numeric_cols].fillna(df_model[numeric_cols].mean())

    # Impute missing categorical values with mode (most frequent value)
    categorical_cols = df_model.select_dtypes(include=[object]).columns
    for col in categorical_cols:
        df_model[col] = df_model[col].fillna(df_model[col].mode()[0])

    print(f"\nData Shape after imputing missing values: {df_model.shape}")

    # Ensure all data is numeric (for any potential conversion issues)
    print("Ensuring all data is numeric...")
    df_model = df_model.apply(pd.to_numeric, errors='coerce')

    # Drop remaining rows with any missing data
    df_model = df_model.dropna()
    print(f"Data Shape after dropping remaining NaNs: {df_model.shape}")

    # Check if DataFrame is empty after dropping rows with NaNs
    if df_model.shape[0] == 0:
        print("The dataset is empty after dropping NaNs. Please check for data issues.")
        return

    # Define target variable and features
    y = df_model['ScaleScore']
    X = df_model.drop(['ScaleScore'], axis=1)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # 1. Correlation Analysis
    print("\nPerforming Correlation Analysis...")
    corr_matrix = df_model.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()
    print("Correlation matrix saved as 'correlation_matrix.png'.")

    # Identify top correlated features with the target variable
    corr_with_target = corr_matrix['ScaleScore'].abs().sort_values(ascending=False)
    print("\nTop features correlated with ScaleScore:")
    print(corr_with_target.head(10))

    # 2. Feature Importance using Random Forest
    print("\nTraining Random Forest Regressor for Feature Importance...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Save the model and feature list
    joblib.dump(rf_model, 'model_deployment/student_performance_model.pkl')
    joblib.dump(X.columns.tolist(), 'model_deployment/model_features.pkl')

    # Feature importance
    importances = rf_model.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Plot Feature Importance
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(20))
    plt.title('Top 20 Feature Importances')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    print("Feature importance plot saved as 'feature_importance.png'.")

    # 3. Model Evaluation
    print("\nEvaluating Random Forest Model...")
    y_pred_rf = rf_model.predict(X_test)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2_rf = r2_score(y_test, y_pred_rf)
    print(f"Random Forest RMSE: {rmse_rf:.2f}")
    print(f"Random Forest R^2 Score: {r2_rf:.2f}")

    # 4. Model Comparison
    print("\nComparing Different Models...")

    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    r2_lr = r2_score(y_test, y_pred_lr)
    print(f"Linear Regression RMSE: {rmse_lr:.2f}")
    print(f"Linear Regression R^2 Score: {r2_lr:.2f}")

    # Gradient Boosting Regressor
    gb_model = GradientBoostingRegressor(random_state=42)
    gb_model.fit(X_train, y_train)
    y_pred_gb = gb_model.predict(X_test)
    rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))
    r2_gb = r2_score(y_test, y_pred_gb)
    print(f"Gradient Boosting RMSE: {rmse_gb:.2f}")
    print(f"Gradient Boosting R^2 Score: {r2_gb:.2f}")

    # Compare RMSE and R^2
    models_evaluation = pd.DataFrame({
        'Model': ['Random Forest', 'Linear Regression', 'Gradient Boosting'],
        'RMSE': [rmse_rf, rmse_lr, rmse_gb],
        'R^2 Score': [r2_rf, r2_lr, r2_gb]
    })
    print("\nModel Evaluation Summary:")
    print(models_evaluation)

    # 5. Clustering Analysis
    print("\nPerforming Enhanced Clustering Analysis...")

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    df_model['Cluster'] = kmeans.fit_predict(X)

    # Investigating clusters
    cluster_counts = df_model['Cluster'].value_counts()
    print(f"\nCluster counts:\n{cluster_counts}")

    # Visualize clusters using PCA for dimensionality reduction
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df_model['Cluster'], palette="Set1", alpha=0.7)
    plt.title('Clusters visualized using PCA (2 components)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.savefig('clusters_pca_visualization.png')
    plt.close()
    print("Cluster visualization saved as 'clusters_pca_visualization.png'.")

    # Optional: Analyze characteristics of each cluster
    top_features = ['ScaleScore', 'AvgScaleScoreByAssessmentSchool', 'AvgScaleScoreBySubjectSchool']
    for cluster in df_model['Cluster'].unique():
        cluster_df = df_model[df_model['Cluster'] == cluster]
        print(f"\nCluster {cluster} summary:")
        print(cluster_df.describe())

        # Visualize the distribution of key features within each cluster
        plt.figure(figsize=(12, 8))
        for feature in top_features:
            sns.histplot(cluster_df[feature], label=feature, kde=True)

        plt.title(f'Distribution of Key Features in Cluster {cluster}')
        plt.legend()
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.savefig(f'cluster_{cluster}_feature_distribution.png')
        plt.close()
        print(f"Cluster {cluster} feature distributions saved as 'cluster_{cluster}_feature_distribution.png'.")

    # Further Analysis of Cluster Characteristics
    for feature in top_features:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Cluster', y=feature, data=df_model)
        plt.title(f'Boxplot of {feature} across Clusters')
        plt.xlabel('Cluster')
        plt.ylabel(f'{feature}')
        plt.savefig(f'{feature}_boxplot_clusters.png')
        plt.close()
        print(f"Boxplot of {feature} across clusters saved as '{feature}_boxplot_clusters.png'.")

    # Display pairplot of top features by cluster
    sns.pairplot(df_model[top_features + ['Cluster']], hue='Cluster', palette='Set1')
    plt.savefig('pairplot_clusters.png')
    plt.close()
    print("Pairplot of clusters saved as 'pairplot_clusters.png'.")

    # 6. Statistical Analysis
    print("\nPerforming Statistical Analysis...")
    gender_columns = [col for col in df_model.columns if 'Gender_' in col]
    if 'Gender_Male' in gender_columns and 'Gender_Female' in gender_columns:
        male_scores = df_model[df_model['Gender_Male'] == 1]['ScaleScore']
        female_scores = df_model[df_model['Gender_Female'] == 1]['ScaleScore']

        # T-test
        t_stat, p_value = stats.ttest_ind(male_scores, female_scores, equal_var=False)
        print(f"T-test between male and female ScaleScores:")
        print(f"T-statistic: {t_stat:.2f}, P-value: {p_value:.4f}")
    else:
        print("Gender columns not found for statistical analysis.")

    # 7. Anomaly Detection
    print("\nPerforming Anomaly Detection...")
    iso_forest = IsolationForest(contamination=0.01, random_state=42)
    df_model['Anomaly'] = iso_forest.fit_predict(X)

    num_anomalies = (df_model['Anomaly'] == -1).sum()
    print(f"Number of anomalies detected: {num_anomalies}")

    # Investigate the anomalies
    anomalies_df = df_model[df_model['Anomaly'] == -1]
    print(f"\nSample anomalies:\n{anomalies_df.head()}")

    # Plot anomalies
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=anomalies_df, x='AvgScaleScoreBySubjectSchool', y='ScaleScore')
    plt.title('Anomalies: ScaleScore vs AvgScaleScoreBySubjectSchool')
    plt.xlabel('Avg Scale Score By Subject School')
    plt.ylabel('Scale Score')
    plt.savefig('anomalies_scatterplot.png')
    plt.close()
    print("Anomalies scatterplot saved as 'anomalies_scatterplot.png'.")

    # 8. Data Visualization
    print("\nCreating Data Visualizations...")
    # Distribution of ScaleScore
    plt.figure(figsize=(8, 6))
    sns.histplot(df_model['ScaleScore'], kde=True)
    plt.title('Distribution of Scale Scores')
    plt.xlabel('Scale Score')
    plt.ylabel('Frequency')
    plt.savefig('scale_score_distribution.png')
    plt.close()
    print("Scale score distribution plot saved as 'scale_score_distribution.png'.")

    # Boxplot of ScaleScore by Grade
    plt.figure(figsize=(12, 8))
    grade_cols = [col for col in df_model.columns if 'Grade_' in col]
    if grade_cols:
        df_model['Grade'] = df_model[grade_cols].idxmax(axis=1).str.replace('Grade_', '')
        sns.boxplot(x='Grade', y='ScaleScore', data=df_model)
        plt.title('Scale Score by Grade')
        plt.xlabel('Grade')
        plt.ylabel('Scale Score')
        plt.savefig('scale_score_by_grade.png')
        plt.close()
        print("Scale score by grade plot saved as 'scale_score_by_grade.png'.")
    else:
        print("Grade columns not found for boxplot.")

    # 9. Predictive Insights
    print("\nGenerating Predictive Insights...")
    X_test_copy = X_test.copy()
    X_test_copy['PredictedScaleScore'] = y_pred_rf
    X_test_copy['ActualScaleScore'] = y_test
    X_test_copy['Residual'] = y_test - y_pred_rf
    lowest_scores = X_test_copy.nsmallest(5, 'PredictedScaleScore')
    print("Students with lowest predicted Scale Scores:")
    print(lowest_scores[['PredictedScaleScore', 'ActualScaleScore', 'Residual']])

    # 10. Save Evaluation Metrics
    models_evaluation.to_csv('model_evaluation_summary.csv', index=False)
    print("Model evaluation summary saved as 'model_evaluation_summary.csv'.")

if __name__ == "__main__":
    train_and_analyze()
