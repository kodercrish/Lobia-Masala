##overallQuality<3
##ParkingCap taken

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error

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
    outlier_mask &= ~((X_train_raw['OverallQuality'] < 3) )

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
    'Id',  'BoundaryFence', 'ExtraFacility', 'ServiceLaneType','PropertyClass' 
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
# --- 4.5. Train/Validation Split ---
print("\n--- 4.5. Creating Train/Validation Split ---")
from sklearn.model_selection import KFold, train_test_split
# We split the FE-applied data (X_train_fe) and both log/original y_train
# This gives us a new, smaller training set (fit) and a validation set (val)
X_train_fit, X_val, y_train_fit_log, y_val_log, y_train_fit, y_val = train_test_split(
    X_train_fe, 
    y_train_log, 
    y_train,        # Include the original y_train for scoring
    test_size=0.15,   # 15% of the data will be for validation
    random_state=42
)

print(f"Training split shape: {X_train_fit.shape}")
print(f"Validation split shape: {X_val.shape}")

# --- 5. Model Training (on 100% Data) ---
print("\n--- 5A. Baseline Linear Regression ---")

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import Ridge

# Baseline Linear Regression pipeline
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train baseline model ON 85% TRAINING SPLIT
print(f"Training baseline model on {X_train_fit.shape[0]} training samples...")
lr_pipeline.fit(X_train_fit, y_train_fit_log)

# --- Evaluate baseline model ON 85% TRAINING SPLIT ---
print(f"Evaluating baseline on {X_train_fit.shape[0]} training samples...")
y_train_fit_log_pred_lr = lr_pipeline.predict(X_train_fit)
y_train_fit_pred_lr = np.expm1(y_train_fit_log_pred_lr)
y_train_fit_pred_lr[y_train_fit_pred_lr < 0] = 0

rmse_train_fit_lr = root_mean_squared_error(y_train_fit, y_train_fit_pred_lr)
r2_train_fit_lr = r2_score(y_train_fit, y_train_fit_pred_lr)

print(f"Baseline Linear Regression RMSE (Train Split): {rmse_train_fit_lr:,.2f}")
print(f"Baseline Linear Regression R² (Train Split): {r2_train_fit_lr:.4f}")

# --- Evaluate baseline model ON 15% VALIDATION DATA ---
print(f"Evaluating baseline on {X_val.shape[0]} VALIDATION samples...")
y_val_log_pred_lr = lr_pipeline.predict(X_val)
y_val_pred_lr = np.expm1(y_val_log_pred_lr)
y_val_pred_lr[y_val_pred_lr < 0] = 0

rmse_val_lr = root_mean_squared_error(y_val, y_val_pred_lr)
r2_val_lr = r2_score(y_val, y_val_pred_lr)

print(f"Baseline Linear Regression RMSE (Validation): {rmse_val_lr:,.2f}")
print(f"Baseline Linear Regression R² (Validation): {r2_val_lr:.4f}")


# --- 5B. GridSearchCV for Linear Regression (on 85% Data) ---
print("\n--- 5B. GridSearchCV for Linear Regression ---")

ridge_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Ridge(random_state=42, max_iter=10000))
])

# Define GridSearchCV parameter grid
param_grid = {
    'regressor__alpha': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
    'regressor__fit_intercept': [True, False],
    'regressor__tol': [1e-4, 1e-3]
}

cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=ridge_pipeline,
    param_grid=param_grid,
    cv=cv_strategy,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

# Train grid search ON 85% TRAINING SPLIT
print(f"Running GridSearchCV on {X_train_fit.shape[0]} training samples...")
grid_search.fit(X_train_fit, y_train_fit_log)

print("\nBest GridSearchCV Hyperparameters (from 85% split):")
print(grid_search.best_params_)

# Get the best model found by the search
# This model is ALREADY FITTED on the 85% (X_train_fit) data
best_gs_model_from_split = grid_search.best_estimator_

# --- Evaluate grid search model ON 85% TRAINING SPLIT ---
print(f"Evaluating GridSearch model on {X_train_fit.shape[0]} training samples...")
y_train_fit_log_pred_gs = best_gs_model_from_split.predict(X_train_fit)
y_train_fit_pred_gs = np.expm1(y_train_fit_log_pred_gs)
y_train_fit_pred_gs[y_train_fit_pred_gs < 0] = 0

rmse_train_fit_gs = root_mean_squared_error(y_train_fit, y_train_fit_pred_gs)
r2_train_fit_gs = r2_score(y_train_fit, y_train_fit_pred_gs)

print(f"GridSearch Optimized RMSE (Train Split): {rmse_train_fit_gs:,.2f}")
print(f"GridSearch Optimized R² (Train Split): {r2_train_fit_gs:.4f}")

# --- Evaluate grid search model ON 15% VALIDATION DATA ---
print(f"Evaluating GridSearch model on {X_val.shape[0]} VALIDATION samples...")
y_val_log_pred_gs = best_gs_model_from_split.predict(X_val)
y_val_pred_gs = np.expm1(y_val_log_pred_gs)
y_val_pred_gs[y_val_pred_gs < 0] = 0

rmse_val_gs = root_mean_squared_error(y_val, y_val_pred_gs)
r2_val_gs = r2_score(y_val, y_val_pred_gs)

print(f"GridSearch Optimized RMSE (Validation): {rmse_val_gs:,.2f}")
print(f"GridSearch Optimized R² (Validation): {r2_val_gs:.4f}")


# --- 6. Final Model Selection (Based on Validation Score) ---
print("\n--- 6. Final Model Selection ---")
print("--- Validation Set Performance Comparison ---")
print(f"Baseline Linear Regression RMSE (Validation): {rmse_val_lr:,.2f} | R²: {r2_val_lr:.4f}")
print(f"GridSearch Optimized Regression RMSE (Validation): {rmse_val_gs:,.2f} | R²: {r2_val_gs:.4f}")

# Decide which model pipeline to use based on VALIDATION score
if rmse_val_gs < rmse_val_lr:
    print("\n✅ GridSearch model had better validation score. Re-training on 100% of data.")
    # We take the best *pipeline structure* from the grid search
    # and re-fit it on the FULL training dataset (X_train_fe)
    final_model = best_gs_model_from_split.fit(X_train_fe, y_train_log)
else:
    print("\n✅ Baseline Linear Regression had better validation score. Re-training on 100% of data.")
    # We take the baseline pipeline structure
    # and re-fit it on the FULL training dataset (X_train_fe)
    final_model = lr_pipeline.fit(X_train_fe, y_train_log)


# --- 7. Display Final Features ---
print("\n--- 7. Final Features Considered by the Model ---")
try:
    # Access the 'preprocessor' step from the final fitted pipeline
    preprocessor_step = final_model.named_steps['preprocessor']
    
    # Get the feature names out
    final_feature_names = preprocessor_step.get_feature_names_out()
    
    print(f"Total number of features after preprocessing: {len(final_feature_names)}")
    print("List of all features fed into the regressor:")
    
    # Print all feature names
    for i, name in enumerate(final_feature_names):
        print(f"  {i+1}: {name}")

except Exception as e:
    print(f"Could not retrieve feature names: {e}")
# --- END OF NEW SECTION ---


# --- 8. Prediction and Submission File Creation ---
print("\n--- 8. Prediction & Submission ---")
# Predict using the final model that was re-trained on 100% of the data
y_test_log_pred = final_model.predict(X_test_fe)

# Reverse log-transformation
y_test_pred = np.expm1(y_test_log_pred)
y_test_pred[y_test_pred < 0] = 0 # Final check

submission_df = pd.DataFrame({
    'Id': test_ids_cleaned,
    TARGET_COLUMN: y_test_pred
})

submission_filename = 'linear_GridSearchCV_split.csv' # Changed filename
submission_df.to_csv(submission_filename, index=False)

print("Prediction process complete.")
print(f"Submission file '{submission_filename}' created with {len(submission_df)} predictions.")
print("First 5 test predictions:")
print(submission_df.head())