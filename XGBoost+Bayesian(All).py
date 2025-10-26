##overallQuality<3
##ParkingCap taken

import pandas as pd
import numpy as np
# --- NEW/CHANGED IMPORTS ---
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
# --- END NEW/CHANGED IMPORTS ---
from xgboost import XGBRegressor  # <-- CHANGED
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error, r2_score
import warnings
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
# --- ADD THESE LINES ---
from skopt import BayesSearchCV
from skopt.space import Real, Integer
# --- END OF ADDED LINES ---
from sklearn.compose import ColumnTransformer
# Suppress warnings
warnings.filterwarnings('ignore')

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
        raise

# Separate features and target before dropping/cleaning
X_train_raw = df_train.drop(columns=[TARGET_COLUMN])
y_train_raw = df_train[TARGET_COLUMN]
X_test = df_test.copy()
test_ids = X_test['Id']

print(f"Initial training data shape: {X_train_raw.shape}")


# --- 1.5 Outlier Removal (New Step) ---
initial_row_count = len(df_train)

# 1. Target-based cleaning
y_lower_bound = y_train_raw.quantile(0.001)
y_upper_bound = y_train_raw.quantile(0.999)
outlier_mask = (y_train_raw >= y_lower_bound) & (y_train_raw <= y_upper_bound)

# 2. Predictor-based cleaning
if 'UsableArea' in X_train_raw.columns:
    outlier_mask &= (X_train_raw['UsableArea'] < 4000)
if 'OverallQuality' in X_train_raw.columns and 'UsableArea' in X_train_raw.columns:
    outlier_mask &= ~((X_train_raw['OverallQuality'] < 3) & (X_train_raw['UsableArea'] > 3000))

# Apply the mask
X_train = X_train_raw[outlier_mask].copy()
y_train = y_train_raw[outlier_mask].copy()
test_ids_cleaned = X_test['Id'] 

print(f"Rows removed due to extreme outliers: {initial_row_count - len(X_train)}")


# --- 1.6 Advanced Feature Merging (Before Dropping Columns) ---
print("\nMerging basement features...")
basement_quality_map = {
    'GLQ': 5, 'ALQ': 4, 'BLQ': 3, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'None': 0
}
for df in [X_train, X_test]:
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

# Feature Engineering for Pool
print("Engineering Pool features...")
pool_quality_map = { 'None': 0, 'Fa': 1, 'Ex': 2 }
for df in [X_train, X_test]:
    df['SwimmingPoolArea'] = df['SwimmingPoolArea'].fillna(0)
    df['PoolQuality'] = df['PoolQuality'].fillna('None')
    df['PoolQuality_Score'] = df['PoolQuality'].map(pool_quality_map).fillna(0)
    df['TotalPoolScore'] = df['PoolQuality_Score'] * df['SwimmingPoolArea']
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

# Columns to drop
columns_to_drop = [
    'Id',  'BoundaryFence', 'ExtraFacility', 'ServiceLaneType', 
    'BasementHalfBaths', 'LowQualityArea','FacadeType',
    'ParkingArea',  # Keep ParkingCapacity
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
    
    # Define the standard 5-point quality map
    quality_map_5pt = {
        'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0
    }
    
    # --- ORDINAL PARKING MAPPING ---
    parking_finish_map = { 'Fin': 3, 'RFn': 2, 'Unf': 1, 'None': 0 }
    df['ParkingQuality'] = df['ParkingQuality'].fillna('None').map(quality_map_5pt).fillna(0)
    df['ParkingCondition'] = df['ParkingCondition'].fillna('None').map(quality_map_5pt).fillna(0)
    df['ParkingFinish'] = df['ParkingFinish'].fillna('None').map(parking_finish_map).fillna(0)
    
    # --- LOUNGE QUALITY MAPPING ---
    df['LoungeQuality'] = df['LoungeQuality'].fillna('None').map(quality_map_5pt).fillna(0)
    
    # --- PROPERTY FUNCTIONALITY MAPPING ---
    functionality_map = {
        'Typ': 7, 'Min1': 6, 'Min2': 5, 'Mod': 4, 
        'Maj1': 3, 'Maj2': 2, 'Sev': 1, 'None': 0
    }
    df['PropertyFunctionality'] = df['PropertyFunctionality'].fillna('None').map(functionality_map).fillna(0)
    
    # --- EXTERIOR QUALITY/CONDITION MAPPING ---
    df['ExteriorQuality'] = df['ExteriorQuality'].fillna('None').map(quality_map_5pt).fillna(0)
    df['ExteriorCondition'] = df['ExteriorCondition'].fillna('None').map(quality_map_5pt).fillna(0)
    
    # --- BASEMENT FEATURES MAPPING ---
    df['BasementHeight'] = df['BasementHeight'].fillna('None').map(quality_map_5pt).fillna(0)
    df['BasementCondition'] = df['BasementCondition'].fillna('None').map(quality_map_5pt).fillna(0)
    exposure_map = { 'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'None': 0 }
    df['BasementExposure'] = df['BasementExposure'].fillna('None').map(exposure_map).fillna(0)

    # --- KITCHEN/HEATING QUALITY MAPPING ---
    df['KitchenQuality'] = df['KitchenQuality'].fillna('None').map(quality_map_5pt).fillna(0)
    df['HeatingQuality'] = df['HeatingQuality'].fillna('None').map(quality_map_5pt).fillna(0)

    # --- Time-based features ---
    df['HouseAge'] = df['YearSold'] - df['ConstructionYear']
    df['RenovationYear'] = df['RenovationYear'].fillna(df['ConstructionYear'])
    df.loc[df['RenovationYear'] == 0, 'RenovationYear'] = df.loc[df['RenovationYear'] == 0, 'ConstructionYear']
    df['YearsSinceModification'] = df['YearSold'] - df[['ConstructionYear', 'RenovationYear']].max(axis=1)
    
    # --- Interaction features ---
    df['QualityArea'] = df['OverallQuality'] * df['UsableArea']
    if 'FullBaths' in df.columns and 'HalfBaths' in df.columns:
        df['TotalBathrooms'] = df['FullBaths'] + (0.5 * df['HalfBaths'])
    
    # --- Log transformation ---
    for col in ['RoadAccessLength', 'LandArea', 'FacadeArea', 'BasementTotalSF', 'ParkingArea']:
        if col in df.columns:
            temp_df = df[col].fillna(0)
            df[col + '_Log'] = np.log1p(temp_df)
    
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

# --- 4. Preprocessing Pipelines (MODIFIED FOR TREE MODELS) ---

# Numerical Transformer: Impute only. NO SCALING.
tree_numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# Categorical Transformer: Impute and ORDINAL ENCODE.
tree_categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='None')), 
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# Create the final preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', tree_numerical_transformer, numerical_features),
        ('cat', tree_categorical_transformer, categorical_features)
    ],
    remainder='drop'
)

# --- 5. Model Training (on 100% Data) ---
print("\n--- 5A. Baseline XGBoost ---")

from xgboost import XGBRegressor  # Make sure this import is at the top
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import KFold
from skopt import BayesSearchCV
from skopt.space import Real, Integer

# Baseline XGBoost pipeline (with default parameters)
xgb_baseline_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(n_estimators=100, 
                                objective='reg:squarederror', 
                                eval_metric='rmse', 
                                random_state=42, 
                                n_jobs=-1))
])

# Train baseline model ON 100% TRAINING DATA
print("Training baseline model on 100% training data...")
xgb_baseline_pipeline.fit(X_train_fe, y_train_log)

# --- Evaluate baseline model ON 100% TRAINING DATA ---
print("Evaluating baseline on 100% training data...")
y_train_log_pred_xgb_base = xgb_baseline_pipeline.predict(X_train_fe)
y_train_pred_xgb_base = np.expm1(y_train_log_pred_xgb_base)
y_train_pred_xgb_base[y_train_pred_xgb_base < 0] = 0

rmse_train_xgb_base = root_mean_squared_error(y_train, y_train_pred_xgb_base)
r2_train_xgb_base = r2_score(y_train, y_train_pred_xgb_base)

print(f"Baseline XGBoost RMSE (Train): {rmse_train_xgb_base:,.2f}")
print(f"Baseline XGBoost R² (Train): {r2_train_xgb_base:.4f}")


# --- 5B. Bayesian Optimization for XGBoost (on 100% Data) ---
print("\n--- 5B. Bayesian Optimization for XGBoost ---")

xgb_bayes_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(objective='reg:squarederror', 
                                eval_metric='rmse', 
                                random_state=42, 
                                n_jobs=-1))
])

# Define Bayesian search space for XGBoost
search_space = {
    'regressor__n_estimators': Integer(100, 1000),
    'regressor__learning_rate': Real(0.01, 0.3, prior='log-uniform'),
    'regressor__max_depth': Integer(3, 10),
    'regressor__subsample': Real(0.7, 1.0, prior='uniform'),
    'regressor__colsample_bytree': Real(0.7, 1.0, prior='uniform'),
    'regressor__reg_alpha': Real(0.01, 10.0, prior='log-uniform'), # L1 reg
    'regressor__reg_lambda': Real(0.01, 10.0, prior='log-uniform')  # L2 reg
}

cv_strategy = KFold(n_splits=3, shuffle=True, random_state=42)

bayes_search = BayesSearchCV(
    estimator=xgb_bayes_pipeline,
    search_spaces=search_space,
    n_iter=30,
    cv=cv_strategy,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    verbose=2,
    random_state=42
)

# Train bayesian search ON 100% TRAINING DATA
print("Running Bayesian Search on 100% training data...")
bayes_search.fit(X_train_fe, y_train_log)

print("\nBest Bayesian Hyperparameters:")
print(bayes_search.best_params_)

# Get the best model found by the search
best_bayes_model = bayes_search.best_estimator_

# --- Evaluate bayesian model ON 100% TRAINING DATA ---
print("Evaluating Bayesian model on 100% training data...")
y_train_log_pred_bayes = best_bayes_model.predict(X_train_fe)
y_train_pred_bayes = np.expm1(y_train_log_pred_bayes)
y_train_pred_bayes[y_train_pred_bayes < 0] = 0

rmse_train_bayes = root_mean_squared_error(y_train, y_train_pred_bayes)
r2_train_bayes = r2_score(y_train, y_train_pred_bayes)

print(f"Bayesian Optimized RMSE (Train): {rmse_train_bayes:,.2f}")
print(f"Bayesian Optimized R² (Train): {r2_train_bayes:.4f}")


# --- 6. Final Model Selection (Based on Training Score) ---
print("\n--- 6. Final Model Selection ---")
print("--- Training Set Performance Comparison ---")
print(f"Baseline XGBoost RMSE (Train): {rmse_train_xgb_base:,.2f} | R²: {r2_train_xgb_base:.4f}")
print(f"Bayesian Optimized Regression RMSE (Train): {rmse_train_bayes:,.2f} | R²: {r2_train_bayes:.4f}")

# Decide which model pipeline to use based on TRAINING score
if rmse_train_bayes < rmse_train_xgb_base:
    print("\n✅ Bayesian-optimized model had better training score. Using as final model.")
    final_model = best_bayes_model
else:
    print("\n✅ Baseline XGBoost had better training score. Using as final model.")
    final_model = xgb_baseline_pipeline


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
y_test_log_pred = final_model.predict(X_test_fe)

# Reverse log-transformation
y_test_pred = np.expm1(y_test_log_pred)
y_test_pred[y_test_pred < 0] = 0 # Final check

submission_df = pd.DataFrame({
    'Id': test_ids_cleaned,
    TARGET_COLUMN: y_test_pred
})

submission_filename = 'TestModels/xgb_bayes_fulltrain_submission.csv' # Changed filename
submission_df.to_csv(submission_filename, index=False)

print("Prediction process complete.")
print(f"Submission file '{submission_filename}' created with {len(submission_df)} predictions.")
print("First 5 test predictions:")
print(submission_df.head())