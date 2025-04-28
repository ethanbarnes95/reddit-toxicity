# train_xgboost_smotetomek_fixed.py
"""
Train XGBoost model with Optuna hyperparameter tuning,
using SMOTETomek combined sampling (FIXED) to handle class imbalance.
Includes threshold evaluation, feature importance analysis,
final performance visualizations (Confusion Matrix, PR Curve),
and learning curve generation (Loss, Accuracy, AUC PR).

Uses both numeric features and pre-computed user embeddings.

Expects data split Parquets in config.SPLIT_DIR:
  - train.parquet
  - valid.parquet
  - test.parquet
"""

import os
import joblib
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay,
                             precision_recall_curve, PrecisionRecallDisplay,
                             precision_score, recall_score)
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE # <-- Add this import
from imblearn.combine import SMOTETomek
# Assuming config.py defines EMBEDDING_DIM, SPLIT_DIR, MODEL_DIR, RANDOM_SEED
import config

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING) # Reduce Optuna verbosity

# --- Configuration ---
NUMERIC_COLS = ["n_posts", "mean_score"]
# Ensure EMBEDDING_DIM is correctly defined in config.py or set manually
EMBEDDING_DIM = getattr(config, 'EMBEDDING_DIM', 384) # Default to 384 if not in config
N_TRIALS = 30 # Number of Optuna trials
PLOT_OUTPUT_DIR = "eda_plots_fixed" # Use a new dir for fixed results
MODEL_OUTPUT_PATH = f"{config.MODEL_DIR}/xgb_user_tox_smotetomek_fixed.joblib"
SCALER_OUTPUT_PATH = f"{config.MODEL_DIR}/scaler_xgb_smotetomek_fixed.joblib"
# --- SET OPTIMAL THRESHOLD BASED ON ANALYSIS (Update after re-running) ---
OPTIMAL_THRESHOLD = 0.30 # Defaulting based on last run, may change
LEARNING_CURVE_METRIC = 'aucpr' # Metric for Optuna and one of the learning curves

# --- Create output directories if they don't exist ---
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
os.makedirs(config.MODEL_DIR, exist_ok=True)


# --- Helper Function to Load Data and Prepare Features ---
def load_and_prepare_data(split_name):
    """Loads Parquet split, extracts features, and handles embeddings."""
    print(f"Loading {split_name}.parquet...")
    file_path = f"{config.SPLIT_DIR}/{split_name}.parquet"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}. Please ensure data splitting is done.")
    df = pd.read_parquet(file_path)
    print(f"Columns in {split_name}.parquet: {df.columns.tolist()}")

    required_cols = NUMERIC_COLS + ["emb", "weak_label"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in {split_name}.parquet: {missing_cols}")

    X_numeric = df[NUMERIC_COLS].values.astype(np.float32)
    if 'emb' not in df.columns:
         raise ValueError(f"'emb' column not found in {split_name}.parquet")
    try:
        X_emb = np.array(df["emb"].tolist(), dtype=np.float32)
    except Exception as e:
        raise ValueError(f"Error converting 'emb' column to numpy array in {split_name}.parquet: {e}")

    if X_emb.ndim != 2 or X_emb.shape[1] != EMBEDDING_DIM:
         raise ValueError(f"Expected embedding dim {EMBEDDING_DIM}, got shape {X_emb.shape} in {split_name}.parquet")
    y = df["weak_label"].values
    print(f"Successfully loaded {len(df)} rows from {split_name}.parquet.")
    return X_numeric, X_emb, y

# --- 1. Load Data ---
print("Loading data splits...")
try:
    X_tr_numeric, X_tr_emb, y_tr = load_and_prepare_data("train")
    X_va_numeric, X_va_emb, y_va = load_and_prepare_data("valid")
    X_te_numeric, X_te_emb, y_te = load_and_prepare_data("test")
    print(f"Train shapes: Numeric {X_tr_numeric.shape}, Emb {X_tr_emb.shape}, y {y_tr.shape}")
    print(f"Valid shapes: Numeric {X_va_numeric.shape}, Emb {X_va_emb.shape}, y {y_va.shape}")
    print(f"Test shapes:  Numeric {X_te_numeric.shape}, Emb {X_te_emb.shape}, y {y_te.shape}")
except (FileNotFoundError, ValueError) as e:
     print(f"Error during data loading: {e}")
     exit()


# --- 2. Scale Numeric Features ---
print("Scaling numeric features...")
scaler = StandardScaler()
X_tr_numeric_scaled = scaler.fit_transform(X_tr_numeric)
X_va_numeric_scaled = scaler.transform(X_va_numeric)
X_te_numeric_scaled = scaler.transform(X_te_numeric)

# --- 3. Combine Features ---
print("Combining features...")
X_tr_combined = np.hstack((X_tr_numeric_scaled, X_tr_emb))
X_va_combined = np.hstack((X_va_numeric_scaled, X_va_emb)) # Renamed for clarity
X_te_combined = np.hstack((X_te_numeric_scaled, X_te_emb)) # Renamed for clarity
print(f"Combined feature shapes before sampling: Train {X_tr_combined.shape}, Valid {X_va_combined.shape}, Test {X_te_combined.shape}")
print(f"Original training class distribution:\n{pd.Series(y_tr).value_counts(normalize=True)}")

# --- 4. Apply SMOTETomek Sampling ---
print("\nApplying SMOTETomek sampling to training data...")
n_minority = np.sum(y_tr == 1)
# k_neighbors must be less than the number of minority samples
smote_neighbors = min(5, n_minority - 1) if n_minority > 1 else 1

if smote_neighbors < 1:
    print("Warning: Minority class has less than 2 samples. Skipping SMOTETomek.")
    X_tr_final = X_tr_combined
    y_tr_final = y_tr
else:
    try:
        # Define the SMOTE part explicitly
        # *** FIX: Removed n_jobs=-1 as it's not supported in older imblearn versions for SMOTE ***
        smote_sampler = SMOTE(random_state=config.RANDOM_SEED, k_neighbors=smote_neighbors)
        # Pass the SMOTE object to SMOTETomek
        # Note: n_jobs applies to the TomekLinks part here. This is usually supported.
        smt = SMOTETomek(random_state=config.RANDOM_SEED, smote=smote_sampler, n_jobs=-1)

        X_tr_resampled, y_tr_resampled = smt.fit_resample(X_tr_combined, y_tr)
        print(f"Resampled training shapes: X {X_tr_resampled.shape}, y {y_tr_resampled.shape}")
        print(f"Resampled training class distribution:\n{pd.Series(y_tr_resampled).value_counts(normalize=True)}")
        X_tr_final = X_tr_resampled
        y_tr_final = y_tr_resampled
    except Exception as e:
        # Keep the error print statement, but ensure the assignment happens
        print(f"Error during SMOTETomek: {e}. Using original training data.")
        X_tr_final = X_tr_combined
        y_tr_final = y_tr


# --- 5. Optuna Objective Function ---
# Using the resampled data (X_tr_final, y_tr_final) for training inside the objective
# Using the original validation data (X_va_combined, y_va) for evaluation
def objective(trial):
    """Optuna objective function to find best XGBoost hyperparameters."""
    param = {
        'objective': 'binary:logistic', 'eval_metric': LEARNING_CURVE_METRIC,
        'eta': trial.suggest_float('learning_rate', 1e-4, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'random_state': config.RANDOM_SEED, 'tree_method': 'hist'
    }
    xgb_model = xgb.XGBClassifier(**param)
    xgb_model.fit(X_tr_final, y_tr_final, # Train on (potentially) resampled data
                  eval_set=[(X_va_combined, y_va)], # Evaluate on original validation data
                  early_stopping_rounds=50,
                  verbose=False)
    preds_proba = xgb_model.predict_proba(X_va_combined)[:, 1]
    ap_score = average_precision_score(y_va, preds_proba)
    return ap_score

# --- 6. Run Optuna Study ---
print(f"\nStarting Optuna study ({N_TRIALS} trials)...")
study = optuna.create_study(direction='maximize', study_name='xgboost_toxicity_smotetomek_fixed')
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
print(f"\nStudy finished!")
print("Best AP (validation) during study:", study.best_value)
print("Best params found by Optuna:", study.best_params)

# --- 7. Train Final Model & Capture Learning Curve Data ---
print("\nTraining final model with best parameters...")
final_params = study.best_params
final_params['objective'] = 'binary:logistic'
final_params['random_state'] = config.RANDOM_SEED
final_params['eval_metric'] = list(set([LEARNING_CURVE_METRIC, 'logloss', 'error']))

final_model = xgb.XGBClassifier(**final_params)
eval_set = [(X_tr_final, y_tr_final), (X_va_combined, y_va)] # Use combined validation features
final_model.fit(X_tr_final, y_tr_final,
                eval_set=eval_set,
                verbose=False)
results = final_model.evals_result()

# --- 8. Evaluate on Test Set (Predictions) ---
print("\nEvaluating final model on test set...")
test_pred_proba = final_model.predict_proba(X_te_combined)[:, 1] # Use combined test features
test_ap = average_precision_score(y_te, test_pred_proba)
print(f"\nXGBoost SMOTETomek (Fixed) Test AP: {test_ap:.3f}")

# --- 9. Report for Optimal Threshold ---
print(f"\n--- Test Classification Report (Optimal Threshold: {OPTIMAL_THRESHOLD:.2f}) ---")
test_pred_labels_optimal = (test_pred_proba > OPTIMAL_THRESHOLD).astype(int)
print(classification_report(y_te, test_pred_labels_optimal, target_names=['non-toxic (0)', 'toxic (1)'], digits=3, zero_division=0))

# --- 10. Evaluate different thresholds (for analysis) ---
print("\n--- Evaluating different thresholds on Test Set (for analysis) ---")
thresholds = [0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1]
f1_scores = {} # Store F1 scores to find the best
for thresh in thresholds:
    print(f"\n--- Threshold: {thresh:.2f} ---")
    test_pred_labels_thresh = (test_pred_proba > thresh).astype(int)
    p_thresh = precision_score(y_te, test_pred_labels_thresh, pos_label=1, zero_division=0)
    r_thresh = recall_score(y_te, test_pred_labels_thresh, pos_label=1, zero_division=0)
    f1 = 2 * (p_thresh * r_thresh) / (p_thresh + r_thresh) if (p_thresh + r_thresh) > 0 else 0
    f1_scores[thresh] = f1
    print(f"  Precision: {p_thresh:.3f}")
    print(f"  Recall:    {r_thresh:.3f}")
    print(f"  F1-Score:  {f1:.3f}")
    # print(classification_report(y_te, test_pred_labels_thresh, target_names=['non-toxic (0)', 'toxic (1)'], digits=3, zero_division=0))

# Find and print best threshold based on F1
best_f1_thresh = max(f1_scores, key=f1_scores.get)
print(f"\nBest threshold based on F1 Score: {best_f1_thresh:.2f} (F1 = {f1_scores[best_f1_thresh]:.3f})")
print(f"(Note: OPTIMAL_THRESHOLD variable is currently set to {OPTIMAL_THRESHOLD:.2f})")


# --- 11. Save Model and Scaler ---
print("\nSaving model and scaler...")
joblib.dump(final_model, MODEL_OUTPUT_PATH, compress=3)
joblib.dump(scaler, SCALER_OUTPUT_PATH, compress=3)
print(f"✓ Saved model -> {MODEL_OUTPUT_PATH}")
print(f"✓ Saved scaler -> {SCALER_OUTPUT_PATH}")

# --- 12. Feature Importance Analysis ---
# (Code remains the same as before)
print("\n--- Calculating and Plotting Feature Importance ---")
try:
    emb_feature_names = [f'emb_{i}' for i in range(EMBEDDING_DIM)]
    numeric_cols_str = [str(col) for col in NUMERIC_COLS]
    all_feature_names = numeric_cols_str + emb_feature_names
    if len(all_feature_names) != final_model.n_features_in_:
         print(f"Warning: Feature name mismatch. Using generic names.")
         all_feature_names = [f'f{i}' for i in range(final_model.n_features_in_)]
    importance_scores = final_model.feature_importances_
    feature_importance_df = pd.DataFrame({'feature': all_feature_names, 'importance': importance_scores})
    top_n = 20
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
    top_features_df = feature_importance_df.head(top_n)
    print(f"\nTop {top_n} Features:")
    print(top_features_df)
    plt.figure(figsize=(10, max(6, top_n * 0.4)))
    sns.barplot(x='importance', y='feature', data=top_features_df, palette='viridis', orient='h')
    plt.title(f'Top {top_n} Feature Importances (XGBoost SMOTETomek Fixed Model)')
    plt.xlabel('Importance Score (gain)')
    plt.ylabel('Feature')
    plt.tight_layout()
    importance_plot_path = os.path.join(PLOT_OUTPUT_DIR, "feature_importance_top20_fixed.png")
    plt.savefig(importance_plot_path)
    print(f"\nSaved feature importance plot to {importance_plot_path}")
    plt.close()
except AttributeError:
    print("Could not access feature_importances_. Skipping feature importance plot.")
except Exception as e:
    print(f"Error during feature importance calculation or plotting: {e}")


# --- 13. Performance Visualizations ---
# (Code remains the same, but uses updated variables like test_ap, test_pred_proba, test_pred_labels_optimal)
print("\n--- Generating Performance Visualizations ---")

# a) Confusion Matrix (using OPTIMAL_THRESHOLD set at top)
try:
    cm = confusion_matrix(y_te, test_pred_labels_optimal)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-toxic (0)', 'Toxic (1)'])
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap='Blues', ax=ax, values_format='d')
    ax.set_title(f'Confusion Matrix (Threshold = {OPTIMAL_THRESHOLD:.2f})')
    plt.tight_layout()
    cm_path = os.path.join(PLOT_OUTPUT_DIR, "confusion_matrix_optimal_fixed.png")
    plt.savefig(cm_path)
    print(f"Saved Confusion Matrix plot to {cm_path}")
    plt.close()
except Exception as e:
    print(f"Error during Confusion Matrix plotting: {e}")


# b) Precision-Recall Curve
try:
    fig, ax = plt.subplots(figsize=(8, 7))
    pr_display = PrecisionRecallDisplay.from_predictions(
        y_te, test_pred_proba, ax=ax, name='XGBoost SMOTETomek (Fixed)', color='darkorange'
    )
    ax.set_title(f'Precision-Recall Curve (Test AP = {test_ap:.3f})')
    p_opt = precision_score(y_te, test_pred_labels_optimal, pos_label=1, zero_division=0)
    r_opt = recall_score(y_te, test_pred_labels_optimal, pos_label=1, zero_division=0)
    ax.plot(r_opt, p_opt, 'o', markersize=8, label=f'Operating Point (Thresh={OPTIMAL_THRESHOLD:.2f})\nP={p_opt:.2f}, R={r_opt:.2f}', color='red', zorder=10)
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    pr_curve_path = os.path.join(PLOT_OUTPUT_DIR, "precision_recall_curve_fixed.png")
    plt.savefig(pr_curve_path)
    print(f"Saved Precision-Recall Curve plot to {pr_curve_path}")
    plt.close()
except Exception as e:
    print(f"Error during Precision-Recall Curve plotting: {e}")

# c) Learning Curves (Loss, Accuracy/Error, and original metric)
print("\n--- Generating Learning Curves ---")
# Plot Loss (LogLoss)
try:
    if 'logloss' in results['validation_0'] and 'logloss' in results['validation_1']:
        epochs = len(results['validation_0']['logloss'])
        x_axis = range(0, epochs)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x_axis, results['validation_0']['logloss'], label='Train Loss (LogLoss)')
        ax.plot(x_axis, results['validation_1']['logloss'], label='Validation Loss (LogLoss)')
        ax.legend()
        plt.ylabel('LogLoss')
        plt.xlabel('Boosting Rounds')
        plt.title('XGBoost Training vs. Validation Loss')
        plt.grid(True, linestyle='--', alpha=0.6)
        loss_curve_path = os.path.join(PLOT_OUTPUT_DIR, "learning_curve_loss_fixed.png")
        plt.savefig(loss_curve_path)
        print(f"Saved Loss Learning Curve plot to {loss_curve_path}")
        plt.close()
    else: print("Could not find 'logloss' in evaluation results. Skipping loss plot.")
except Exception as e: print(f"Error during Loss Learning Curve plotting: {e}.")

# Plot Accuracy (using 1 - error)
try:
    if 'error' in results['validation_0'] and 'error' in results['validation_1']:
        epochs = len(results['validation_0']['error'])
        x_axis = range(0, epochs)
        train_accuracy = 1.0 - np.array(results['validation_0']['error'])
        val_accuracy = 1.0 - np.array(results['validation_1']['error'])
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x_axis, train_accuracy, label='Train Accuracy')
        ax.plot(x_axis, val_accuracy, label='Validation Accuracy')
        ax.legend()
        plt.ylabel('Accuracy')
        plt.xlabel('Boosting Rounds')
        plt.title('XGBoost Training vs. Validation Accuracy')
        plt.ylim(max(0, np.min(val_accuracy)-0.1), 1.05)
        plt.grid(True, linestyle='--', alpha=0.6)
        accuracy_curve_path = os.path.join(PLOT_OUTPUT_DIR, "learning_curve_accuracy_fixed.png")
        plt.savefig(accuracy_curve_path)
        print(f"Saved Accuracy Learning Curve plot to {accuracy_curve_path}")
        plt.close()
    else: print("Could not find 'error' in evaluation results. Skipping accuracy plot.")
except Exception as e: print(f"Error during Accuracy Learning Curve plotting: {e}.")

# Plot Original Metric (e.g., AUC PR)
if LEARNING_CURVE_METRIC not in ['logloss', 'error']:
    try:
        if LEARNING_CURVE_METRIC in results['validation_0'] and LEARNING_CURVE_METRIC in results['validation_1']:
            epochs = len(results['validation_0'][LEARNING_CURVE_METRIC])
            x_axis = range(0, epochs)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(x_axis, results['validation_0'][LEARNING_CURVE_METRIC], label=f'Train ({LEARNING_CURVE_METRIC.upper()})')
            ax.plot(x_axis, results['validation_1'][LEARNING_CURVE_METRIC], label=f'Validation ({LEARNING_CURVE_METRIC.upper()})')
            ax.legend()
            plt.ylabel(f'{LEARNING_CURVE_METRIC.upper()} Score')
            plt.xlabel('Boosting Rounds')
            plt.title(f'XGBoost Learning Curve ({LEARNING_CURVE_METRIC.upper()})')
            plt.grid(True, linestyle='--', alpha=0.6)
            learning_curve_path = os.path.join(PLOT_OUTPUT_DIR, f"learning_curve_{LEARNING_CURVE_METRIC}_fixed.png")
            plt.savefig(learning_curve_path)
            print(f"Saved {LEARNING_CURVE_METRIC.upper()} Learning Curve plot to {learning_curve_path}")
            plt.close()
        else: print(f"Could not find '{LEARNING_CURVE_METRIC}' in evaluation results. Skipping this plot.")
    except Exception as e: print(f"Error during {LEARNING_CURVE_METRIC.upper()} Learning Curve plotting: {e}.")


print("\nScript finished.")
