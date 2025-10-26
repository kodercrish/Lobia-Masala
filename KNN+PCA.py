##overallQuality<3
##ParkingCap taken

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV,BayesianRidge
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, GridSearchCV

# Define file paths
train_data_path = 'train.csv'
test_data_path = 'test.csv'
TARGET_COLUMN = 'HotelValue'

# --- 1. Data Loading and Initial Setup ---
print("--- 1. Data Loading ---")
try:
    # Attempt to load from a common nested folder structure
    df_train = pd.read_csv('Hotel-Property-Value-Dataset/train.csv')
    df_test = pd.read_csv('Hotel-Property-Value-Dataset/test.csv')
    print("Loaded data from 'Hotel-Property-Value-Dataset/' folder.")
except FileNotFoundError:
    # Fallback to the current directory
    try:
        df_train = pd.read_csv(train_data_path)
        df_test = pd.read_csv(test_data_path)
        print("Loaded data from current directory.")
    except FileNotFoundError as e:
        print(f"Error: Files not found. Ensure 'train.csv' and 'test.csv' are in the correct location.")
        print(f"Details: {e}")
        # Exit or raise error if data can't be loaded
        raise

# Separate features and target before dropping/cleaning
X_train_raw = df_train.drop(columns=[TARGET_COLUMN])
y_train_raw = df_train[TARGET_COLUMN]
X_test = df_test.copy()
test_ids = X_test['Id']

print(f"Initial training data shape: {X_train_raw.shape}")


# --- 1.5 Outlier Removal (New Step) ---
# Remove samples based on Target Value (extremely low/high values)
# and based on large/extreme values in key predictor columns (UsableArea and OverallQuality).
initial_row_count = len(df_train)

# 1. Target-based cleaning: Remove extreme values (e.g., bottom 0.1% and top 0.1% of prices)
y_lower_bound = y_train_raw.quantile(0.001)
y_upper_bound = y_train_raw.quantile(0.999)
outlier_mask = (y_train_raw >= y_lower_bound) & (y_train_raw <= y_upper_bound)

# 2. Predictor-based cleaning (Common for this type of dataset)
# Remove properties with extremely large UsableArea (e.g., > 4000 sq ft)
if 'UsableArea' in X_train_raw.columns:
    outlier_mask &= (X_train_raw['UsableArea'] < 4000)


# Remove properties with poor OverallQuality and high UsableArea (often errors)
if 'OverallQuality' in X_train_raw.columns and 'UsableArea' in X_train_raw.columns:
    outlier_mask &= ~((X_train_raw['OverallQuality'] < 3) & (X_train_raw['UsableArea'] > 3000))

# Apply the mask to both features and target
X_train = X_train_raw[outlier_mask].copy()
y_train = y_train_raw[outlier_mask].copy()

# Sync test_ids for the remaining rows
test_ids_cleaned = X_test['Id'] # No change to test IDs as we don't drop test rows

print(f"Rows removed due to extreme outliers: {initial_row_count - len(X_train)}")


# --- 1.6 Advanced Feature Merging (Before Dropping Columns) ---
# Merge Basement Features into Weighted Quality Score
print("\nMerging basement features...")
basement_quality_map = {
    'GLQ': 5, 'ALQ': 4, 'BLQ': 3, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'None': 0
}

for df in [X_train, X_test]:
    # Fill NaN values
    df['BasementFacilitySF1'] = df['BasementFacilitySF1'].fillna(0)
    df['BasementFacilitySF2'] = df['BasementFacilitySF2'].fillna(0)
    
    # Map types to scores
    df['Type1_Score'] = df['BasementFacilityType1'].fillna('None').map(basement_quality_map).fillna(0)
    df['Type2_Score'] = df['BasementFacilityType2'].fillna('None').map(basement_quality_map).fillna(0)
    
    # Calculate weighted quality score
    df['TotalBasementScore'] = (df['Type1_Score'] * df['BasementFacilitySF1']) + (df['Type2_Score'] * df['BasementFacilitySF2'])
    df['BasementFinishedSF'] = df['BasementFacilitySF1'] + df['BasementFacilitySF2']
    
    # Average basement quality (cast to float to avoid dtype warning)
    df['BasementAvgQuality'] = 0.0
    mask_has_finished = df['BasementFinishedSF'] > 0
    df.loc[mask_has_finished, 'BasementAvgQuality'] = (
        df.loc[mask_has_finished, 'TotalBasementScore'] / df.loc[mask_has_finished, 'BasementFinishedSF']
    )
    
    # Drop original basement facility columns
    df.drop(columns=['BasementFacilityType1', 'BasementFacilityType2', 
                     'BasementFacilitySF1', 'BasementFacilitySF2',
                     'Type1_Score', 'Type2_Score'], errors='ignore', inplace=True)


# --- NEW SECTION: Feature Engineering for Pool ---
print("Engineering Pool features...")
# Define a quality map for PoolQuality. 
# 'None' (or NaN) = 0, 'Fa' (Fair) = 1, 'Ex' (Excellent) = 4.
# Added 'TA' (Typical) and 'Gd' (Good) as they are common.
pool_quality_map = {
    'None': 0,
    'Fa': 1,
    'Ex': 2,
}

for df in [X_train, X_test]:
    # Fill NaN values first. 'PoolArea' NaNs mean 0 area.
    df['SwimmingPoolArea'] = df['SwimmingPoolArea'].fillna(0)
    df['PoolQuality'] = df['PoolQuality'].fillna('None')
    
    # Map quality strings to numeric scores
    df['PoolQuality_Score'] = df['PoolQuality'].map(pool_quality_map).fillna(0)
    
    # Create the new feature by multiplying quality by area
    df['TotalPoolScore'] = df['PoolQuality_Score'] * df['SwimmingPoolArea']
    
    # Now drop the original columns since they are combined
    df.drop(columns=['PoolQuality', 'SwimmingPoolArea','PoolQuality_Score'],
            errors='ignore', inplace=True)
    
# Merge Porch/Veranda Features
print("Merging porch features...")
for df in [X_train, X_test]:
    df['TotalPorchArea'] = (
        df['OpenVerandaArea'].fillna(0)+
        df['EnclosedVerandaArea'].fillna(0) + 
        df['SeasonalPorchArea'].fillna(0) + 
        df['ScreenPorchArea'].fillna(0)
    )
    df.drop(columns=['OpenVerandaArea','EnclosedVerandaArea', 'SeasonalPorchArea', 'ScreenPorchArea'], 
            errors='ignore', inplace=True)
# --- END OF NEW SECTION ---
# Columns to drop (including multicollinearity removal)
columns_to_drop = [
    'Id',  'BoundaryFence', 'ExtraFacility', 'ServiceLaneType', 
    'BasementHalfBaths', 'LowQualityArea','FacadeType'
    # Multicollinearity removal
    # 'ParkingArea',  # Keep ParkingCapacity
    # 'GroundFloorArea',  # Keep UsableArea
    # 'TotalRooms',       # Keep FullBaths
    # 'UpperFloorArea',   # Captured in UsableArea
]
X_train = X_train.drop(columns=columns_to_drop, errors='ignore')
X_test = X_test.drop(columns=columns_to_drop, errors='ignore')

print(f"Cleaned training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# --- 2. Target Transformation ---
y_train_log = np.log1p(y_train)


# --- 3. Feature Engineering ---
def engineer_features(df):
    df = df.copy()
    
# --- NEW ORDINAL PARKING MAPPING ---
    # Define maps for ordinal parking features
    quality_map_5pt = {
        'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0
    }
    parking_finish_map = {
        'Fin': 3, 'RFn': 2, 'Unf': 1, 'None': 0
    }
    
    # Overwrite categorical columns with their new numerical scores
    
    # Impute and map Quality
    df['ParkingQuality'] = df['ParkingQuality'].fillna('None').map(quality_map_5pt).fillna(0)
    
    # Impute and map Condition
    df['ParkingCondition'] = df['ParkingCondition'].fillna('None').map(quality_map_5pt).fillna(0)
    
    # Impute and map Finish
    df['ParkingFinish'] = df['ParkingFinish'].fillna('None').map(parking_finish_map).fillna(0)
    

    # --- NEW PROPERTY FUNCTIONALITY MAPPING ---
    # This feature represents deductions from 'Typical'
    functionality_map = {
        'Typ': 7,  # Typical
        'Min1': 6, # Minor Deductions 1
        'Min2': 5, # Minor Deductions 2
        'Mod': 4,  # Moderate Deductions
        'Maj1': 3, # Major Deductions 1
        'Maj2': 2, # Major Deductions 2
        'Sev': 1,  # Severely Damaged
        'None': 0  # Assuming 'None' is worse than 'Sev' or not applicable
    }
    ##--- NEW EXTERIOR QUALITY/CONDITION MAPPING ---
    df['ExteriorQuality'] = df['ExteriorQuality'].fillna('None').map(quality_map_5pt).fillna(0)
    df['ExteriorCondition'] = df['ExteriorCondition'].fillna('None').map(quality_map_5pt).fillna(0)
    
    # # --- NEW BASEMENT FEATURES MAPPING ---
    # # BasementHeight (uses 5-point map)
    # df['BasementHeight'] = df['BasementHeight'].fillna('None').map(quality_map_5pt).fillna(0)
    
    # BasementCondition (uses 5-point map)
    df['BasementCondition'] = df['BasementCondition'].fillna('None').map(quality_map_5pt).fillna(0)
    
    # BasementExposure (custom map)
    exposure_map = {
        'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'None': 0
    }
    df['BasementExposure'] = df['BasementExposure'].fillna('None').map(exposure_map).fillna(0)
    

    # --- NEW KITCHEN/HEATING QUALITY MAPPING ---
    df['KitchenQuality'] = df['KitchenQuality'].fillna('None').map(quality_map_5pt).fillna(0)
    df['HeatingQuality'] = df['HeatingQuality'].fillna('None').map(quality_map_5pt).fillna(0)
    # --- END NEW KITCHEN/HEATING SECTION ---

    # Impute and map PropertyFunctionality. 
    # Use fillna('Typ') if 'None' should be treated as 'Typical'
    df['PropertyFunctionality'] = df['PropertyFunctionality'].fillna('None').map(functionality_map).fillna(0)
    # --- END NEW FUNCTIONALITY SECTION ---
    # By overwriting the columns, they will now be automatically
    # treated as 'numerical' features by the rest of the script.
    # Time-based features
    df['HouseAge'] = df['YearSold'] - df['ConstructionYear']
    
    # Handle RenovationYear: if 0 or missing, use ConstructionYear
    df['RenovationYear'] = df['RenovationYear'].fillna(df['ConstructionYear'])
    df.loc[df['RenovationYear'] == 0, 'RenovationYear'] = df.loc[df['RenovationYear'] == 0, 'ConstructionYear']
    
    df['YearsSinceModification'] = df['YearSold'] - df[['ConstructionYear', 'RenovationYear']].max(axis=1)
   ########################################################### 
    # # Interaction features
    # df['QualityArea'] = df['OverallQuality'] * df['UsableArea']
    
    # Bathroom quality feature
    # if 'FullBaths' in df.columns and 'HalfBaths' in df.columns:
    #     df['TotalBathrooms'] = df['FullBaths'] + (0.5 * df['HalfBaths'])
    #########################################################
    
    # Log transformation for skewed numerical features
    for col in ['ParkingConstructionYear','BasementFinishedSF','YearsSinceModification','LandArea', 'BasementTotalSF', 'ParkingArea']:
        if col in df.columns:
            temp_df = df[col].fillna(0)
            df[col + '_Log'] = np.log1p(temp_df)
    df=df.drop(columns=['ParkingConstructionYear','BasementFinishedSF', 'YearsSinceModification','LandArea', 'BasementTotalSF', 'ParkingArea'],errors='ignore')
    
    # Drop source columns used for feature engineering
    df = df.drop(columns=['ConstructionYear', 'RenovationYear', 'YearSold', 'MonthSold'], errors='ignore')
    return df

X_train_fe = engineer_features(X_train)
X_test_fe = engineer_features(X_test)

numerical_features = X_train_fe.select_dtypes(include=np.number).columns.tolist()
categorical_features = X_train_fe.select_dtypes(include=['object', 'category']).columns.tolist()

print("\n--- 3. Feature Engineering Complete ---")
print(f"Number of numerical features: {len(numerical_features)}")
print(f"Number of categorical features: {len(categorical_features)}")
print(f"Final training features shape: {X_train_fe.shape}")

# --- 4. Preprocessing Pipelines ---

# Numerical Transformer: Impute, Scale, and add Polynomial Features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()), 
    # The degree=2 poly features were commented out in your previous code to speed things up. 
    # I'll keep them commented unless performance is a concern.
    # ('poly', PolynomialFeatures(degree=2, include_bias=False)) 
])


# Categorical Transformer: Impute and One-Hot Encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='None')), 
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop'
)

# --- 5. Model Training: KNN with PCA + GridSearchCV ---
print("\n--- 5. Model Training: KNN with PCA + GridSearchCV ---")

# Create the full pipeline including preprocessing, PCA, and KNN
knn_pca_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('pca', PCA()), # PCA step added after preprocessing
    ('regressor', KNeighborsRegressor(n_jobs=-1)) # KNN Regressor
])

# Define the parameter grid for GridSearchCV
# Search over PCA components and KNN parameters
param_grid = {
    'pca__n_components': [0.95, 0.99, 50, 100], # Keep 95%/99% variance, or fixed number
    'regressor__n_neighbors': [3, 5, 7, 10],   # Number of neighbors
    'regressor__weights': ['uniform', 'distance'], # Weighting strategy
    'regressor__metric': ['minkowski', 'manhattan'] # Distance metric
}

# Use KFold for cross-validation within GridSearchCV
cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)

# Setup GridSearchCV
# n_jobs=-1 uses all available CPU cores
grid_search = GridSearchCV(
    estimator=knn_pca_pipeline,
    param_grid=param_grid,
    cv=cv_strategy,
    scoring='neg_root_mean_squared_error', # Optimize for RMSE
    n_jobs=-1,
    verbose=2 # Show progress
)

# Train GridSearchCV on the full training data
print("Running GridSearchCV for KNN with PCA...")
grid_search.fit(X_train_fe, y_train_log)

# --- Results ---
print("\nBest GridSearchCV Hyperparameters found:")
print(grid_search.best_params_)
print(f"\nBest CV Score (Negative RMSE): {grid_search.best_score_:.4f}")

# The best model found by the search IS the final model
final_model = grid_search.best_estimator_
print("\n✅ Using Best KNN+PCA model found by GridSearchCV as final model.")

# Evaluate the final model on the training data (for comparison/sanity check)
print("\nEvaluating final model on training data...")
y_train_log_pred_knn = final_model.predict(X_train_fe)
y_train_pred_knn = np.expm1(y_train_log_pred_knn)
y_train_pred_knn[y_train_pred_knn < 0] = 0

rmse_train_knn = root_mean_squared_error(y_train, y_train_pred_knn)
r2_train_knn = r2_score(y_train, y_train_pred_knn)

print(f"Final KNN+PCA Model RMSE (Train): {rmse_train_knn:,.2f}")
print(f"Final KNN+PCA Model R² (Train): {r2_train_knn:.4f}")

# Print details about the PCA step in the final model
try:
    final_pca = final_model.named_steps['pca']
    print(f"\nPCA Details from Final Model:")
    print(f"  Number of components selected: {final_pca.n_components_}")
    print(f"  Explained variance ratio sum: {final_pca.explained_variance_ratio_.sum():.4f}")
except Exception as e:
    print(f"Could not retrieve PCA details: {e}")

# Print details about the KNN step in the final model
try:
    final_knn = final_model.named_steps['regressor']
    print(f"\nKNN Details from Final Model:")
    print(f"  n_neighbors: {final_knn.n_neighbors}")
    print(f"  weights: {final_knn.weights}")
    print(f"  metric: {final_knn.metric}")
except Exception as e:
    print(f"Could not retrieve KNN details: {e}")

# --- 6. Prediction and Submission File Creation ---
print("\n--- 6. Prediction & Submission ---")
y_test_log_pred = final_model.predict(X_test_fe)

# Reverse log-transformation
y_test_pred = np.expm1(y_test_log_pred)
y_test_pred[y_test_pred < 0] = 0 # Final check to ensure non-negative values

submission_df = pd.DataFrame({
    'Id': test_ids_cleaned,
    TARGET_COLUMN: y_test_pred
})

submission_filename = 'knn_pca.csv' # Changed filename to reflect cleaning
submission_df.to_csv(submission_filename, index=False)

print("Prediction process complete.")
print(f"Submission file '{submission_filename}' created with {len(submission_df)} predictions.")
print("First 5 test predictions:")
print(submission_df.head())