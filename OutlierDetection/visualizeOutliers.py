##overallQuality<3
##ParkingCap taken

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures # Keep for potential future use if needed
from sklearn.compose import ColumnTransformer # Keep for potential future use if needed
from sklearn.pipeline import Pipeline # Keep for potential future use if needed
from sklearn.impute import SimpleImputer # Keep for potential future use if needed
# Import metrics if you want to calculate anything later
# from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
# Define file paths for RAW data
train_data_path = 'train.csv'
# test_data_path = 'test.csv' # Test data not needed for EDA on train
TARGET_COLUMN = 'HotelValue'
TARGET_COLUMN_LOG = 'HotelValue_Log' # Name for the transformed target

# 2. Get the directory to save plots
output_dir = input("Enter the full path of the directory to save the plots: ").strip()
# Create the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
print(f"Plots will be saved in: {output_dir}")

# --- 1. Data Loading and Initial Setup ---
print("\n--- 1. Data Loading (Raw Train Data) ---")
try:
    # Attempt to load from a common nested folder structure
    df_train_raw = pd.read_csv('Hotel-Property-Value-Dataset/train.csv')
    # df_test_raw = pd.read_csv('Hotel-Property-Value-Dataset/test.csv') # Load if needed
    print("Loaded raw data from 'Hotel-Property-Value-Dataset/' folder.")
except FileNotFoundError:
    # Fallback to the current directory
    try:
        df_train_raw = pd.read_csv(train_data_path)
        # df_test_raw = pd.read_csv(test_data_path) # Load if needed
        print("Loaded raw data from current directory.")
    except FileNotFoundError as e:
        print(f"Error: Raw files not found. Ensure '{train_data_path}' is in the correct location.")
        print(f"Details: {e}")
        raise
except Exception as e:
     print(f"An error occurred loading the CSV: {e}")
     raise


# Separate features and target before dropping/cleaning
X_train = df_train_raw.drop(columns=[TARGET_COLUMN], errors='ignore')
y_train = df_train_raw[TARGET_COLUMN]
# X_test_raw = df_test_raw.copy() # Process test if needed later



# --- 3. Feature Merging & Initial Dropping ---
print("\n--- 3. Feature Merging & Initial Dropping ---")
basement_quality_map = {
    'GLQ': 5, 'ALQ': 4, 'BLQ': 3, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'None': 0
}
# Apply merging only to X_train for EDA plots (apply to X_test if needed later)
df_list = [X_train] # Only process train data for EDA

# Merge Basement
print("Merging basement features...")
for df in df_list:
    df['BasementFacilitySF1'] = df['BasementFacilitySF1'].fillna(0)
    df['BasementFacilitySF2'] = df['BasementFacilitySF2'].fillna(0)
    df['Type1_Score'] = df['BasementFacilityType1'].fillna('None').map(basement_quality_map).fillna(0)
    df['Type2_Score'] = df['BasementFacilityType2'].fillna('None').map(basement_quality_map).fillna(0)
    df['TotalBasementScore'] = (df['Type1_Score'] * df['BasementFacilitySF1']) + (df['Type2_Score'] * df['BasementFacilitySF2'])
    df['BasementFinishedSF'] = df['BasementFacilitySF1'] + df['BasementFacilitySF2']
    df['BasementAvgQuality'] = 0.0
    mask_has_finished = df['BasementFinishedSF'] > 0
    df.loc[mask_has_finished, 'BasementAvgQuality'] = (
        df.loc[mask_has_finished, 'TotalBasementScore'] / df.loc[mask_has_finished, 'BasementFinishedSF']
    )
    df.drop(columns=['BasementFacilityType1', 'BasementFacilityType2',
                     'BasementFacilitySF1', 'BasementFacilitySF2',
                     'Type1_Score', 'Type2_Score'], errors='ignore', inplace=True)

# Merge Pool
print("Engineering Pool features...")
pool_quality_map = { 'None': 0, 'Fa': 1, 'Ex': 2 }
for df in df_list:
    df['SwimmingPoolArea'] = df['SwimmingPoolArea'].fillna(0)
    df['PoolQuality'] = df['PoolQuality'].fillna('None')
    df['PoolQuality_Score'] = df['PoolQuality'].map(pool_quality_map).fillna(0)
    df['TotalPoolScore'] = df['PoolQuality_Score'] * df['SwimmingPoolArea']
    df.drop(columns=['PoolQuality', 'SwimmingPoolArea','PoolQuality_Score'],
            errors='ignore', inplace=True)

# Merge Porch
print("Merging porch features...")
for df in df_list:
    df['TotalPorchArea'] = (
        df['OpenVerandaArea'].fillna(0)+
        df['EnclosedVerandaArea'].fillna(0) +
        df['SeasonalPorchArea'].fillna(0) +
        df['ScreenPorchArea'].fillna(0)
    )
    df.drop(columns=['OpenVerandaArea','EnclosedVerandaArea', 'SeasonalPorchArea', 'ScreenPorchArea'],
            errors='ignore', inplace=True)

# Initial Column Dropping
print("Dropping specified columns...")
columns_to_drop = [
    'Id',  'BoundaryFence', 'ExtraFacility', 'ServiceLaneType',
    'BasementHalfBaths', 'LowQualityArea','FacadeType'
    # Keep ParkingArea for now if needed for log transform later
]
# Drop only from X_train for EDA
X_train = X_train.drop(columns=columns_to_drop, errors='ignore')

print(f"Training data shape after merging/dropping: {X_train.shape}")


# --- 4. Target Transformation ---
print("\n--- 4. Target Transformation ---")
y_train_log = np.log1p(y_train)
print("Applied log1p transformation to target variable.")


# --- 5. Feature Engineering Function ---
print("\n--- 5. Feature Engineering ---")
def engineer_features(df):
    df = df.copy()
    quality_map_5pt = { 'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0 }
    parking_finish_map = { 'Fin': 3, 'RFn': 2, 'Unf': 1, 'None': 0 }
    functionality_map = { 'Typ': 7, 'Min1': 6, 'Min2': 5, 'Mod': 4, 'Maj1': 3, 'Maj2': 2, 'Sev': 1, 'None': 0 }
    exposure_map = { 'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'None': 0 }

    # Ordinal Mappings
    df['ParkingQuality'] = df['ParkingQuality'].fillna('None').map(quality_map_5pt).fillna(0)
    df['ParkingCondition'] = df['ParkingCondition'].fillna('None').map(quality_map_5pt).fillna(0)
    df['ParkingFinish'] = df['ParkingFinish'].fillna('None').map(parking_finish_map).fillna(0)
    df['ExteriorQuality'] = df['ExteriorQuality'].fillna('None').map(quality_map_5pt).fillna(0)
    df['ExteriorCondition'] = df['ExteriorCondition'].fillna('None').map(quality_map_5pt).fillna(0)
    df['BasementCondition'] = df['BasementCondition'].fillna('None').map(quality_map_5pt).fillna(0)
    df['BasementExposure'] = df['BasementExposure'].fillna('None').map(exposure_map).fillna(0)
    df['KitchenQuality'] = df['KitchenQuality'].fillna('None').map(quality_map_5pt).fillna(0)
    df['HeatingQuality'] = df['HeatingQuality'].fillna('None').map(quality_map_5pt).fillna(0)
    df['PropertyFunctionality'] = df['PropertyFunctionality'].fillna('None').map(functionality_map).fillna(0)
    # df['BasementHeight'] = df['BasementHeight'].fillna('None').map(quality_map_5pt).fillna(0) # Keep original?

    # Time-based features
    if 'YearSold' in df.columns and 'ConstructionYear' in df.columns:
        df['HouseAge'] = df['YearSold'] - df['ConstructionYear']
    if 'RenovationYear' in df.columns and 'ConstructionYear' in df.columns and 'YearSold' in df.columns:
        df['RenovationYear'] = df['RenovationYear'].fillna(df['ConstructionYear'])
        df.loc[df['RenovationYear'] == 0, 'RenovationYear'] = df.loc[df['RenovationYear'] == 0, 'ConstructionYear']
        df['YearsSinceModification'] = df['YearSold'] - df[['ConstructionYear', 'RenovationYear']].max(axis=1)

    # Log transformation for skewed numerical features & drop originals
    log_cols = ['ParkingConstructionYear','BasementFinishedSF','YearsSinceModification','LandArea', 'BasementTotalSF', 'ParkingArea']
    for col in log_cols:
        if col in df.columns:
            temp_df = df[col].fillna(0)
            df[col + '_Log'] = np.log1p(temp_df)
    df = df.drop(columns=log_cols, errors='ignore')

    # Drop source columns used for feature engineering
    df = df.drop(columns=['ConstructionYear', 'RenovationYear', 'YearSold', 'MonthSold'], errors='ignore')
    return df

# Apply feature engineering to X_train
X_train_fe = engineer_features(X_train)
print("Applied feature engineering steps.")
print(f"Final training features shape for EDA: {X_train_fe.shape}")

# --- 6. Prepare DataFrame for Plotting ---
print("\n--- 6. Preparing Data for Plotting ---")
# Combine processed features and transformed target
df_processed_eda = X_train_fe.copy()
# Ensure index alignment after outlier removal
df_processed_eda[TARGET_COLUMN_LOG] = y_train_log.reset_index(drop=True)

# Identify final numerical and categorical features AFTER engineering
FINAL_NUMERICAL_FEATURES = df_processed_eda.select_dtypes(include=np.number).columns.tolist()
# Exclude the target variable itself from the list of features to plot against
if TARGET_COLUMN_LOG in FINAL_NUMERICAL_FEATURES:
    FINAL_NUMERICAL_FEATURES.remove(TARGET_COLUMN_LOG)

FINAL_CATEGORICAL_FEATURES = df_processed_eda.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"Identified {len(FINAL_NUMERICAL_FEATURES)} numerical features for plotting.")
print(f"Identified {len(FINAL_CATEGORICAL_FEATURES)} categorical features for plotting.")

# --- 7. Generate and Save Scatter Plots (Processed Numerical Features) ---
print(f"\nGenerating and saving Scatter Plots...")
for feature in FINAL_NUMERICAL_FEATURES:
    if feature not in df_processed_eda.columns:
        print(f"Warning: Numerical feature '{feature}' not in DataFrame. Skipping.")
        continue

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_processed_eda, x=feature, y=TARGET_COLUMN_LOG) # Use log target
    plt.title(f'Scatter Plot: {feature} vs. {TARGET_COLUMN_LOG}')
    plt.xlabel(feature)
    plt.ylabel(TARGET_COLUMN_LOG)

    # Define the save path and save the figure
    file_name = f"POSTPROCESS_{feature}_scatter.png" # Add prefix
    save_path = os.path.join(output_dir, file_name)
    try:
        plt.savefig(save_path)
    except Exception as e:
        print(f"Error saving plot {save_path}: {e}")
    plt.close() # Close the plot

print("✅ Scatter plots saved.")


# --- 8. Generate and Save Box Plots (Processed Categorical Features) ---
print(f"\nGenerating and saving Box Plots...")
for feature in FINAL_CATEGORICAL_FEATURES:
    if feature not in df_processed_eda.columns:
        print(f"Warning: Categorical feature '{feature}' not in DataFrame. Skipping.")
        continue

    # Skip features with too many unique categories
    unique_count = df_processed_eda[feature].nunique()
    if unique_count > 50:
        print(f"Skipping '{feature}': Too many unique categories ({unique_count}).")
        continue
    if unique_count <= 1:
        print(f"Skipping '{feature}': Only one unique category.")
        continue


    plt.figure(figsize=(12, 7))
    try:
        # Order boxes by median value
        order = df_processed_eda.groupby(feature)[TARGET_COLUMN_LOG].median().sort_values().index
        sns.boxplot(data=df_processed_eda, x=feature, y=TARGET_COLUMN_LOG, order=order) # Use log target
        plt.title(f'Box Plot: {feature} vs. {TARGET_COLUMN_LOG}')
        plt.xlabel(feature)
        plt.ylabel(TARGET_COLUMN_LOG)
        plt.xticks(rotation=45, ha='right') # Adjust rotation and alignment
        plt.tight_layout() # Adjust layout

        # Define the save path and save the figure
        file_name = f"POSTPROCESS_{feature}_boxplot.png" # Add prefix
        save_path = os.path.join(output_dir, file_name)
        plt.savefig(save_path)
    except Exception as e:
         print(f"Error generating or saving boxplot for {feature}: {e}")
    finally:
        plt.close() # Close the plot

print("✅ Box plots saved.")
print(f"\nAll visualizations have been successfully saved to '{output_dir}'.")