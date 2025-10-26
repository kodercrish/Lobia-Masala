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
    outlier_mask &= ~((X_train_raw['OverallQuality'] < 5) & (X_train_raw['UsableArea'] > 3000))

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

# Merge Porch/Veranda Features
print("Merging porch features...")
for df in [X_train, X_test]:
    df['TotalPorchArea'] = (
        df['OpenVerandaArea'].fillna(0)+
        df['EnclosedVerandaArea'].fillna(0) + 
        df['SeasonalPorchArea'].fillna(0) + 
        df['ScreenPorchArea'].fillna(0)
    
    )
    df.drop(columns=['EnclosedVerandaArea', 'SeasonalPorchArea', 'ScreenPorchArea'], 
            errors='ignore', inplace=True)

# Columns to drop (including multicollinearity removal)
columns_to_drop = [
    'Id', 'PoolQuality', 'BoundaryFence', 'ExtraFacility', 'ServiceLaneType', 
    'BasementHalfBaths', 'LowQualityArea',
    # Multicollinearity removal
    'ParkingCapacity',  # Keep ParkingArea
    'GroundFloorArea',  # Keep UsableArea
    'TotalRooms',       # Keep FullBaths
    'UpperFloorArea',   # Captured in UsableArea
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
    
    # Time-based features
    df['HouseAge'] = df['YearSold'] - df['ConstructionYear']
    
    # Handle RenovationYear: if 0 or missing, use ConstructionYear
    df['RenovationYear'] = df['RenovationYear'].fillna(df['ConstructionYear'])
    df.loc[df['RenovationYear'] == 0, 'RenovationYear'] = df.loc[df['RenovationYear'] == 0, 'ConstructionYear']
    
    df['YearsSinceModification'] = df['YearSold'] - df[['ConstructionYear', 'RenovationYear']].max(axis=1)
    
    # Interaction features
    df['QualityArea'] = df['OverallQuality'] * df['UsableArea']
    
    # Bathroom quality feature
    if 'FullBaths' in df.columns and 'HalfBaths' in df.columns:
        df['TotalBathrooms'] = df['FullBaths'] + (0.5 * df['HalfBaths'])
    
    # Log transformation for skewed numerical features
    for col in ['RoadAccessLength', 'LandArea', 'FacadeArea', 'BasementTotalSF', 'ParkingArea']:
        if col in df.columns:
            temp_df = df[col].fillna(0)
            df[col + '_Log'] = np.log1p(temp_df)
    
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

# --- 5B. Bayesian Optimization for LightGBM ---
print("\n--- 5B. Bayesian Optimization for LightGBM ---")

from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline

# Define LightGBM regressor with safe defaults
regressor = LGBMRegressor(
    random_state=42,
    boosting_type='gbdt',
    n_jobs=-1,
    force_row_wise=True,        # Prevents multi-threading bugs
    min_gain_to_split=0.0,      # Avoids infinite "no positive gain" loops
    verbose=-1
)

# Build pipeline
lgbm_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', regressor)
])

# Define Bayesian search space
search_space = {
    'regressor__num_leaves': Integer(20, 100),
    'regressor__max_depth': Integer(3, 10),
    'regressor__learning_rate': Real(0.01, 0.2, prior='log-uniform'),
    'regressor__n_estimators': Integer(200, 800),
    'regressor__min_child_samples': Integer(5, 50),
    'regressor__subsample': Real(0.6, 1.0, prior='uniform'),
    'regressor__colsample_bytree': Real(0.6, 1.0, prior='uniform'),
    'regressor__reg_alpha': Real(1e-4, 1.0, prior='log-uniform'),
    'regressor__reg_lambda': Real(1e-4, 1.0, prior='log-uniform')
}

from sklearn.model_selection import KFold
cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)

# Bayesian optimization
from skopt import BayesSearchCV
bayes_search = BayesSearchCV(
    estimator=lgbm_pipeline,
    search_spaces=search_space,
    n_iter=30,
    cv=cv_strategy,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

fit_params = {
    'regressor__eval_set': [(X_val, y_val)],
    'regressor__early_stopping_rounds': 10,
    'regressor__verbose': -1
}

bayes_search.fit(X_train_fe, y_train_log, **fit_params)



print("\nBest Bayesian Hyperparameters:")
print(bayes_search.best_params_)


# 5F. Train Final Model with Best Params
final_lgbm_model = bayes_search_lgbm.best_estimator_
final_lgbm_model.fit(X_train_fe, y_train_log)

# 5G. Evaluate on training data
y_train_log_pred_lgbm = final_lgbm_model.predict(X_train_fe)
y_train_pred_lgbm = np.expm1(y_train_log_pred_lgbm)
y_train_pred_lgbm[y_train_pred_lgbm < 0] = 0

rmse_train_lgbm = mean_squared_error(y_train, y_train_pred_lgbm, squared=False)
r2_train_lgbm = r2_score(y_train, y_train_pred_lgbm)

print(f"Training RMSE (Original Scale): {rmse_train_lgbm:,.2f}")
print(f"Training R²: {r2_train_lgbm:.4f}")

# --- 6. Prediction and Submission File Creation ---
print("\n--- 6. Prediction & Submission ---")

y_test_log_pred_lgbm = final_lgbm_model.predict(X_test_fe)
y_test_pred_lgbm = np.expm1(y_test_log_pred_lgbm)
y_test_pred_lgbm[y_test_pred_lgbm < 0] = 0

submission_lgbm = pd.DataFrame({
    'Id': test_ids_cleaned,
    TARGET_COLUMN: y_test_pred_lgbm
})

submission_filename_lgbm = 'lgbm_bayesian_submission.csv'
submission_lgbm.to_csv(submission_filename_lgbm, index=False)

print(f"Submission file '{submission_filename_lgbm}' created with {len(submission_lgbm)} predictions.")
print("First 5 test predictions:")
print(submission_lgbm.head())
