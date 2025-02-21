import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Function to plot heatmap of top correlated features
def plotHeatMap(df, threshold=10):
    """
    Selects the top k features most correlated with 'price' and plots a heatmap.
    :param df: DataFrame containing the dataset
    :param threshold: Number of top correlated features to select (default: 10)
    """
    if 'price' not in df.columns:
        raise ValueError("DataFrame must contain 'price' column")
    
    ind_heatmap = df.corr().price.abs().sort_values(ascending=False)[:threshold].index
    df_heatmap = df[ind_heatmap]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_heatmap.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Heatmap of Top Correlated Features")
    plt.show()

def performVIF(df, drop=True):
    """
    Performs Variance Inflation Factor (VIF) analysis to detect multicollinearity.
    :param df: DataFrame containing numerical features
    :param drop: Boolean flag to drop the highest VIF feature (default: True)
    :return: Updated DataFrame with multicollinearity reduced
    """
    X = df.select_dtypes(include=['float64', 'int64']).copy()

    # Drop constant or near-constant columns to avoid infinite VIF
    X = X.loc[:, X.var() > 0.0001]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]

    # Handle infinite VIF values
    if np.isinf(vif_data["VIF"]).any():
        print("Warning: Infinite VIF detected! Consider removing highly correlated variables.")
        vif_data = vif_data.replace([np.inf, -np.inf], np.nan).dropna()

    # Find the highest VIF feature
    max_vif = vif_data.loc[vif_data['VIF'].idxmax()]
    print(f"Highest VIF Feature: {max_vif['Feature']} with VIF={max_vif['VIF']:.2f}")

    # Drop the feature if required
    if drop and max_vif['VIF'] > 10:
        df = df.drop(columns=[max_vif['Feature']])
        print(f"Dropping {max_vif['Feature']} due to high VIF")

    print(df.info())
    return df