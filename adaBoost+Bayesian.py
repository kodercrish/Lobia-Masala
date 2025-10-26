import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold

# --- NEW MODEL IMPORTS ---
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from skopt import BayesSearchCV
from skopt.space import Categorical, Real, Integer
# --- END NEW IMPORTS ---


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

    # Time-based features
    df['HouseAge'] = df['YearSold'] - df['ConstructionYear']
    
    # Handle RenovationYear: if 0 or missing, use ConstructionYear
    df['RenovationYear'] = df['RenovationYear'].fillna(df['ConstructionYear'])
    df.loc[df['RenovationYear'] == 0, 'RenovationYear'] = df.loc[df['RenovationYear'] == 0, 'ConstructionYear']
    
    df['YearsSinceModification'] = df['YearSold'] - df[['ConstructionYear', 'RenovationYear']].max(axis=1)

    
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

])


from sklearn.preprocessing import OrdinalEncoder # <-- Add this import at the top

# Categorical Transformer: Impute and Ordinal Encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='None')), 
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop'
)

# #############################################################################
# --- 5. Model Training: AdaBoost Regressionn + Bayesian ---
# #############################################################################
print("\n--- 5A. Baseline AdaBoost Regression ---")

# Baseline AdaBoost Regression pipeline
# Using default parameters (DecisionTreeRegressor(max_depth=3))
ada_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', AdaBoostRegressor(random_state=42))
])

# Train baseline model
ada_pipeline.fit(X_train_fe, y_train_log)

# Evaluate baseline model
y_train_log_pred_ada = ada_pipeline.predict(X_train_fe)
y_train_pred_ada = np.expm1(y_train_log_pred_ada)
y_train_pred_ada[y_train_pred_ada < 0] = 0

rmse_train_ada = root_mean_squared_error(y_train, y_train_pred_ada)
r2_train_ada = r2_score(y_train, y_train_pred_ada)

print(f"AdaBoost Baseline RMSE (Train): {rmse_train_ada:,.2f}")
print(f"AdaBoost Baseline R² (Train): {r2_train_ada:.4f}")


# --- 5B. Bayesian Optimization for AdaBoost Regression ---
print("\n--- 5B. Bayesian Optimization for AdaBoost Regression ---")

# We must define the base estimator to tune its parameters
ada_pipeline_tuned = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', AdaBoostRegressor(estimator=DecisionTreeRegressor(), random_state=42))
])

# Define Bayesian search space for AdaBoost
search_space_ada = {
    'regressor__n_estimators': Integer(50, 500),
    'regressor__learning_rate': Real(1e-3, 1.0, prior='log-uniform'),
    'regressor__loss': Categorical(['linear', 'square', 'exponential']),
    # Tune the max_depth of the weak learner (DecisionTreeRegressor)
    'regressor__estimator__max_depth': Integer(2, 8) 
}

cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)

bayes_search_ada = BayesSearchCV(
    estimator=ada_pipeline_tuned,
    search_spaces=search_space_ada,
    n_iter=30,  # Same number of iterations as your original search
    cv=cv_strategy,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

bayes_search_ada.fit(X_train_fe, y_train_log)

print("\nBest Bayesian Hyperparameters for AdaBoost:")
print(bayes_search_ada.best_params_)

# Evaluate optimized model
best_model_ada = bayes_search_ada.best_estimator_

y_train_log_pred_bayes = best_model_ada.predict(X_train_fe)
y_train_pred_bayes = np.expm1(y_train_log_pred_bayes)
y_train_pred_bayes[y_train_pred_bayes < 0] = 0

rmse_train_bayes_ada = root_mean_squared_error(y_train, y_train_pred_bayes)
r2_train_bayes_ada = r2_score(y_train, y_train_pred_bayes)

print(f"\nBayesian Optimized AdaBoost RMSE (Train): {rmse_train_bayes_ada:,.2f}")
print(f"Bayesian Optimized AdaBoost R² (Train): {r2_train_bayes_ada:.4f}")


# --- Compare and Select Best Model ---
print("\n--- Model Comparison ---")
print(f"Baseline AdaBoost RMSE: {rmse_train_ada:,.2f} | R²: {r2_train_ada:.4f}")
print(f"Bayesian Optimized AdaBoost RMSE: {rmse_train_bayes_ada:,.2f} | R²: {r2_train_bayes_ada:.4f}")

if rmse_train_bayes_ada < rmse_train_ada:
    final_model = best_model_ada
    print("✅ Using Bayesian-optimized AdaBoost as final model.")
else:
    final_model = ada_pipeline
    print("✅ Using baseline AdaBoost as final model (performed better or equal).")

# #############################################################################
# --- END OF MODELING SECTION ---
# #############################################################################


# --- NEW SECTION: Display Final Features ---
print("\n--- Final Features Considered by the Model ---")
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

# --- UPDATED FILENAME ---
submission_filename = 'adaboost+Bayesian.csv' 
submission_df.to_csv(submission_filename, index=False)

print("Prediction process complete.")
print(f"Submission file '{submission_filename}' created with {len(submission_df)} predictions.")
print("First 5 test predictions:")
print(submission_df.head())