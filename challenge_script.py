import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

# =========================
# Load Your Dataset
# =========================

file_path = '/home/twoodard/Desktop/hackerman/server_baseline/pertussis_challenge/train_master_gene_NaN_with_predictors_v4.csv'
data = pd.read_csv(file_path)

# =========================
# Define Columns
# =========================

subject_id_column = 'subject_id'  # Column name for subject IDs
day0_columns = ['IgG.PT.Day0', 'Percent.Monocytes.Day0']
target_columns = ['IgG.PT.Day14', 'Percent.Monocytes.Day1']

# Drop the 'specimen_type' column as per your instruction
data = data.drop(columns=['specimen_type'])

# =========================
# Split Data Based on subject_id
# =========================

# Convert subject_id to string to ensure consistency
data[subject_id_column] = data[subject_id_column].astype(str)

# Extract test data for subject_id 119-172
test_subjects = [str(i) for i in range(119, 173)]  # Adjusted range to include 172
data_test = data[data[subject_id_column].isin(test_subjects)].reset_index(drop=True)
data_train = data[~data[subject_id_column].isin(test_subjects)].reset_index(drop=True)

# =========================
# Prepare Training Data
# =========================

X_train = data_train.drop(columns=[subject_id_column] + day0_columns + target_columns)
y_train = data_train[target_columns]

# Identify numerical features
numerical_features = X_train.select_dtypes(include=[np.number]).columns.tolist()

# Retain only numerical features
X_train = X_train[numerical_features]

# Remove columns with all missing values from X_train
cols_with_all_missing = X_train.columns[X_train.isnull().all()]
X_train = X_train.drop(columns=cols_with_all_missing)

# Store the selected features to apply the same transformations to X_test
selected_features = X_train.columns.tolist()

# Check for NaNs in y_train and drop corresponding rows
# Combine X_train and y_train
train_combined = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)

# Drop rows where target variables have NaN
train_combined = train_combined.dropna(subset=target_columns)

# Separate X_train and y_train after dropping NaNs
X_train = train_combined[selected_features]
y_train = train_combined[target_columns]

# Handle missing values and scaling for training data
imputer = SimpleImputer(strategy='median')
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=selected_features)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)

# =========================
# Prepare Test Data
# =========================

X_test = data_test.drop(columns=[subject_id_column] + day0_columns + target_columns)

# Retain only numerical features
X_test = X_test[numerical_features]

# Ensure that X_test has the same columns as X_train
X_test = X_test[selected_features]

# Handle missing values and scaling for test data
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=selected_features)
X_test_scaled = scaler.transform(X_test_imputed)

# Extract IDs and Day 0 values
subject_id_test = data_test[subject_id_column].values
igg_d0_test = data_test['IgG.PT.Day0'].values
monocyte_d0_test = data_test['Percent.Monocytes.Day0'].values

# =========================
# Define Models and Hyperparameters
# =========================

param_grid_rf = {
    'n_estimators': [100],
    'max_depth': [None],
    'min_samples_split': [2]
}

param_grid_mlp = {
    'hidden_layer_sizes': [(100,)],
    'activation': ['relu'],
    'solver': ['adam'],
    'alpha': [0.0001],
    'learning_rate': ['constant']
}

param_grid_gb = {
    'n_estimators': [100],
    'max_depth': [3],
    'learning_rate': [0.1],
    'subsample': [1.0]
}

rf_igg = RandomForestRegressor(random_state=42)
rf_monocytes = RandomForestRegressor(random_state=42)
mlp_igg = MLPRegressor(random_state=42, max_iter=500)
mlp_monocytes = MLPRegressor(random_state=42, max_iter=500)
gb_igg = GradientBoostingRegressor(random_state=42)
gb_monocytes = GradientBoostingRegressor(random_state=42)

def train_model(model, param_grid, X_train, y_train, model_name):
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring='neg_mean_absolute_error',
        verbose=1,
        error_score='raise'
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    return best_model

# Train models for IgG.PT.Day14
best_rf_igg = train_model(rf_igg, param_grid_rf, X_train_scaled, y_train['IgG.PT.Day14'], 'IgG.PT.Day14 Random Forest')
best_mlp_igg = train_model(mlp_igg, param_grid_mlp, X_train_scaled, y_train['IgG.PT.Day14'], 'IgG.PT.Day14 MLPRegressor')
best_gb_igg = train_model(gb_igg, param_grid_gb, X_train_scaled, y_train['IgG.PT.Day14'], 'IgG.PT.Day14 Gradient Boosting')

# Train models for Percent.Monocytes.Day1
best_rf_monocytes = train_model(rf_monocytes, param_grid_rf, X_train_scaled, y_train['Percent.Monocytes.Day1'], 'Percent.Monocytes.Day1 Random Forest')
best_mlp_monocytes = train_model(mlp_monocytes, param_grid_mlp, X_train_scaled, y_train['Percent.Monocytes.Day1'], 'Percent.Monocytes.Day1 MLPRegressor')
best_gb_monocytes = train_model(gb_monocytes, param_grid_gb, X_train_scaled, y_train['Percent.Monocytes.Day1'], 'Percent.Monocytes.Day1 Gradient Boosting')

# =========================
# Make Predictions on Test Data (subject_id 119-172)
# =========================

# Predict IgG.PT.Day14
igg_predictions_rf = best_rf_igg.predict(X_test_scaled)
igg_predictions_mlp = best_mlp_igg.predict(X_test_scaled)
igg_predictions_gb = best_gb_igg.predict(X_test_scaled)

# Predict Percent.Monocytes.Day1
monocyte_predictions_rf = best_rf_monocytes.predict(X_test_scaled)
monocyte_predictions_mlp = best_mlp_monocytes.predict(X_test_scaled)
monocyte_predictions_gb = best_gb_monocytes.predict(X_test_scaled)

# =========================
# Aggregate Predictions by Subject
# =========================

# Create DataFrames for predictions with specimen-level data
igg_preds_df = pd.DataFrame({
    'subject_id': subject_id_test,
    'IgG.PT.Day0': igg_d0_test,
    'IgG_pred_rf': igg_predictions_rf,
    'IgG_pred_mlp': igg_predictions_mlp,
    'IgG_pred_gb': igg_predictions_gb
})

monocyte_preds_df = pd.DataFrame({
    'subject_id': subject_id_test,
    'Percent.Monocytes.Day0': monocyte_d0_test,
    'Monocyte_pred_rf': monocyte_predictions_rf,
    'Monocyte_pred_mlp': monocyte_predictions_mlp,
    'Monocyte_pred_gb': monocyte_predictions_gb
})

# Aggregate predictions by subject_id (e.g., take the mean)
igg_preds_agg = igg_preds_df.groupby('subject_id').mean().reset_index()
monocyte_preds_agg = monocyte_preds_df.groupby('subject_id').mean().reset_index()

# Compute ranks based on aggregated predictions
for model in ['rf', 'mlp', 'gb']:
    igg_preds_agg[f'IgG_rank_{model}'] = igg_preds_agg[f'IgG_pred_{model}'].rank(ascending=False)
    monocyte_preds_agg[f'Monocyte_rank_{model}'] = monocyte_preds_agg[f'Monocyte_pred_{model}'].rank(ascending=False)

# =========================
# Save Individual Predictions Per Subject
# =========================

# Create a dictionary to map model abbreviations to full names
model_names = {
    'rf': 'Random_Forest',
    'mlp': 'MLPRegressor',
    'gb': 'Gradient_Boosting'
}

# Save individual predictions for IgG.PT.Day14
for model_key, model_name in model_names.items():
    igg_results = igg_preds_agg[['subject_id', 'IgG.PT.Day0', f'IgG_pred_{model_key}', f'IgG_rank_{model_key}']]
    igg_results = igg_results.rename(columns={
        f'IgG_pred_{model_key}': 'IgG_pred',
        f'IgG_rank_{model_key}': 'IgG_rank'
    })
    # Save to CSV and TSV
    igg_results.to_csv(
        f'/home/twoodard/Desktop/hackerman/server_baseline/pertussis_challenge/IgG_pt_day14_predictions_{model_name}.csv',
        index=False
    )
    igg_results.to_csv(
        f'/home/twoodard/Desktop/hackerman/server_baseline/pertussis_challenge/IgG_pt_day14_predictions_{model_name}.tsv',
        sep='\t',
        index=False
    )

# Save individual predictions for Percent.Monocytes.Day1
for model_key, model_name in model_names.items():
    monocyte_results = monocyte_preds_agg[['subject_id', 'Percent.Monocytes.Day0', f'Monocyte_pred_{model_key}', f'Monocyte_rank_{model_key}']]
    monocyte_results = monocyte_results.rename(columns={
        f'Monocyte_pred_{model_key}': 'Monocyte_pred',
        f'Monocyte_rank_{model_key}': 'Monocyte_rank'
    })
    # Save to CSV and TSV
    monocyte_results.to_csv(
        f'/home/twoodard/Desktop/hackerman/server_baseline/pertussis_challenge/Monocyte_d1_predictions_{model_name}.csv',
        index=False
    )
    monocyte_results.to_csv(
        f'/home/twoodard/Desktop/hackerman/server_baseline/pertussis_challenge/Monocyte_d1_predictions_{model_name}.tsv',
        sep='\t',
        index=False
    )

# =========================
# Save Combined Predictions Per Target Variable
# =========================

# For IgG.PT.Day14
igg_combined = igg_preds_agg[['subject_id', 'IgG.PT.Day0']]
for model_key in ['rf', 'mlp', 'gb']:
    igg_combined[f'IgG_pred_{model_key}'] = igg_preds_agg[f'IgG_pred_{model_key}']
    igg_combined[f'IgG_rank_{model_key}'] = igg_preds_agg[f'IgG_rank_{model_key}']

# Save combined predictions for IgG.PT.Day14
igg_combined.to_csv(
    '/home/twoodard/Desktop/hackerman/server_baseline/pertussis_challenge/IgG_pt_day14_predictions_all_models.csv',
    index=False
)
igg_combined.to_csv(
    '/home/twoodard/Desktop/hackerman/server_baseline/pertussis_challenge/IgG_pt_day14_predictions_all_models.tsv',
    sep='\t',
    index=False
)

# For Percent.Monocytes.Day1
monocyte_combined = monocyte_preds_agg[['subject_id', 'Percent.Monocytes.Day0']]
for model_key in ['rf', 'mlp', 'gb']:
    monocyte_combined[f'Monocyte_pred_{model_key}'] = monocyte_preds_agg[f'Monocyte_pred_{model_key}']
    monocyte_combined[f'Monocyte_rank_{model_key}'] = monocyte_preds_agg[f'Monocyte_rank_{model_key}']

# Save combined predictions for Percent.Monocytes.Day1
monocyte_combined.to_csv(
    '/home/twoodard/Desktop/hackerman/server_baseline/pertussis_challenge/Monocyte_d1_predictions_all_models.csv',
    index=False
)
monocyte_combined.to_csv(
    '/home/twoodard/Desktop/hackerman/server_baseline/pertussis_challenge/Monocyte_d1_predictions_all_models.tsv',
    sep='\t',
    index=False
)

print("Aggregated predictions per subject saved successfully.")
print("Combined predictions for all models saved successfully.")
