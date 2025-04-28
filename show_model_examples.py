# show_model_examples.py
"""
Loads the trained XGBoost model, test data, raw posts, and user labels
to display concrete examples of model predictions:
- One True Positive (TP) user with top toxic posts and tox_frac.
- One True Negative (TN) user with their single most toxic post and tox_frac.
- One False Positive (FP) user with top toxic posts and tox_frac.
- One False Negative (FN) user with top toxic posts and tox_frac.
- Top 5 users predicted as toxic by the model, with their most toxic post and tox_frac.
"""

import os
import joblib
import pandas as pd
import numpy as np
import config  # Import project configuration

# --- Configuration ---
MODEL_PATH = f"{config.MODEL_DIR}/xgb_user_tox_smotetomek_fixed.joblib"
SCALER_PATH = f"{config.MODEL_DIR}/scaler_xgb_smotetomek_fixed.joblib"
TEST_DATA_PATH = f"{config.SPLIT_DIR}/test.parquet"
RAW_POSTS_PATH = config.RAW_POSTS_CSV
USER_LABELS_PATH = config.USER_LABELS_CSV # Path to user labels for tox_frac
# Use the threshold determined during training/evaluation
# Found 0.30 in train_xgboost_smotetomek_fixed.py, verify if needed
OPTIMAL_THRESHOLD = 0.30
TOP_N_USERS = 5 # Number of top toxic users to show

# --- Helper Function to Load Data and Prepare Features ---
def load_and_prepare_test_data(test_path, scaler):
    """Loads test Parquet, extracts features, scales, and handles embeddings."""
    print(f"Loading test data from {test_path}...")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found: {test_path}")
    df_test = pd.read_parquet(test_path)

    # Ensure necessary columns for features and labels exist
    required_cols = config.NUMERIC_COLS + ["emb", "weak_label", "username"]
    missing_cols = [col for col in required_cols if col not in df_test.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in {test_path}: {missing_cols}")

    print("Preparing features...")
    X_numeric = df_test[config.NUMERIC_COLS].values.astype(np.float32)
    X_emb = np.array(df_test["emb"].tolist(), dtype=np.float32)

    if X_emb.ndim != 2 or X_emb.shape[1] != config.EMBEDDING_DIM:
         raise ValueError(f"Expected embedding dim {config.EMBEDDING_DIM}, got shape {X_emb.shape}")

    print("Scaling numeric features...")
    X_numeric_scaled = scaler.transform(X_numeric)

    print("Combining features...")
    X_combined = np.hstack((X_numeric_scaled, X_emb))

    print(f"Loaded and prepared {len(df_test)} test users.")
    return df_test, X_combined

# --- Helper Function to Load Raw Posts ---
def load_raw_posts(raw_posts_path):
    """Loads relevant columns from the raw posts CSV."""
    print(f"Loading raw posts from {raw_posts_path}...")
    if not os.path.exists(raw_posts_path):
        raise FileNotFoundError(f"Raw posts data not found: {raw_posts_path}")

    # Load only necessary columns to save memory
    try:
        df_posts = pd.read_csv(
            raw_posts_path,
            usecols=['username', 'id', 'title', 'body', 'tox_score'],
            low_memory=False, # Helps with mixed types if any
            encoding='utf-8' # Specify encoding
        )
        # Handle potential NaN values in text fields
        df_posts['title'] = df_posts['title'].fillna('')
        df_posts['body'] = df_posts['body'].fillna('')
        # Ensure tox_score is numeric, coerce errors to NaN, then fill with 0
        df_posts['tox_score'] = pd.to_numeric(df_posts['tox_score'], errors='coerce').fillna(0.0)

        print(f"Loaded {len(df_posts)} raw post entries.")
        return df_posts
    except Exception as e:
        print(f"Error loading raw posts CSV: {e}")
        raise

# --- Helper Function to Load User Labels ---
def load_user_labels(user_labels_path):
    """Loads user labels CSV to get tox_frac."""
    print(f"Loading user labels from {user_labels_path}...")
    if not os.path.exists(user_labels_path):
        raise FileNotFoundError(f"User labels data not found: {user_labels_path}")
    try:
        df_labels = pd.read_csv(user_labels_path, usecols=['username', 'tox_frac'])
        print(f"Loaded {len(df_labels)} user labels.")
        return df_labels
    except Exception as e:
        print(f"Error loading user labels CSV: {e}")
        raise

# --- Helper Function to Display Top Posts ---
def display_top_posts(df_posts, username, num_posts=1, title="Most Toxic Post"):
    """Fetches and prints the top N toxic posts for a user."""
    print(f"\n  {title} (from posts_raw.csv, sorted by tox_score):")
    user_posts = df_posts[df_posts['username'] == username].sort_values('tox_score', ascending=False)
    if user_posts.empty:
        print("    (No raw posts found for this user in the loaded CSV)")
    else:
        for i, post in user_posts.head(num_posts).iterrows():
            post_text = post['body'] if post['body'] else post['title']
            # Limit display length for readability
            post_text_short = (post_text[:150] + '...') if len(post_text) > 150 else post_text
            print(f"    Post {i+1 if num_posts > 1 else ''}: Score={post['tox_score']:.3f} | Text: \"{post_text_short}\"")
            print(f"           (Post ID: {post['id']})")

# --- Main Execution ---
def main():
    # 1. Load Model, Scaler, Raw Posts, User Labels
    print("Loading model, scaler, raw posts, and user labels...")
    if not os.path.exists(MODEL_PATH): raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not os.path.exists(SCALER_PATH): raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    df_posts_raw = load_raw_posts(RAW_POSTS_PATH)
    df_user_labels = load_user_labels(USER_LABELS_PATH)
    print("All necessary files loaded successfully.")

    # 2. Load and Prepare Test Data
    df_test, X_test_combined = load_and_prepare_test_data(TEST_DATA_PATH, scaler)

    # 3. Make Predictions
    print("Making predictions on the test set...")
    pred_proba = model.predict_proba(X_test_combined)[:, 1]
    pred_labels = (pred_proba >= OPTIMAL_THRESHOLD).astype(int)

    # Add predictions and merge tox_frac to the test dataframe
    df_test['predicted_proba'] = pred_proba
    df_test['predicted_label'] = pred_labels
    df_test = pd.merge(df_test, df_user_labels, on='username', how='left')
    # Fill potential NaN tox_frac if a test user wasn't in user_labels.csv (shouldn't happen with inner join in aggregate_users.py)
    df_test['tox_frac'] = df_test['tox_frac'].fillna(-1.0) # Use -1 to indicate missing

    # 4. Identify Example Cases (TP, TN, FP, FN)
    print("\nIdentifying TP, TN, FP, FN examples...")
    true_positives  = df_test[(df_test['weak_label'] == 1) & (df_test['predicted_label'] == 1)]
    true_negatives  = df_test[(df_test['weak_label'] == 0) & (df_test['predicted_label'] == 0)]
    false_positives = df_test[(df_test['weak_label'] == 0) & (df_test['predicted_label'] == 1)]
    false_negatives = df_test[(df_test['weak_label'] == 1) & (df_test['predicted_label'] == 0)]

    # Select one example for each category (e.g., most confident prediction within category)
    tp_example = true_positives.loc[true_positives['predicted_proba'].idxmax()] if not true_positives.empty else None
    tn_example = true_negatives.loc[true_negatives['predicted_proba'].idxmin()] if not true_negatives.empty else None
    fp_example = false_positives.loc[false_positives['predicted_proba'].idxmax()] if not false_positives.empty else None
    fn_example = false_negatives.loc[false_negatives['predicted_proba'].idxmin()] if not false_negatives.empty else None

    print(f"Selected TP example: User '{tp_example['username'] if tp_example is not None else 'N/A'}'")
    print(f"Selected TN example: User '{tn_example['username'] if tn_example is not None else 'N/A'}'")
    print(f"Selected FP example: User '{fp_example['username'] if fp_example is not None else 'N/A'}'")
    print(f"Selected FN example: User '{fn_example['username'] if fn_example is not None else 'N/A'}'")


    # 5. Display Examples
    print("\n" + "="*60)
    print(" MODEL PREDICTION EXAMPLES (Classification & Misclassification)")
    print("="*60)

    # --- True Positive ---
    if tp_example is not None:
        print("\n--- True Positive (Predicted Toxic [✓], Actually Toxic [✓]) ---")
        print(f"Username:          {tp_example['username']}")
        print(f"N Posts:           {tp_example['n_posts']}")
        print(f"Mean Score:        {tp_example['mean_score']:.2f}")
        print(f"Original Tox Frac: {tp_example['tox_frac']:.3f} (Threshold: {config.USER_TOX_THRESHOLD})")
        print(f"Predicted Prob:    {tp_example['predicted_proba']:.4f} -> Label: {tp_example['predicted_label']}")
        print(f"True Weak Label:   {tp_example['weak_label']}")
        display_top_posts(df_posts_raw, tp_example['username'], num_posts=2, title="Top 2 Most Toxic Posts")
    else:
        print("\n--- True Positive ---")
        print("  (No example found in test set)")

    # --- True Negative ---
    if tn_example is not None:
        print("\n--- True Negative (Predicted Non-Toxic [✓], Actually Non-Toxic [✓]) ---")
        print(f"Username:          {tn_example['username']}")
        print(f"N Posts:           {tn_example['n_posts']}")
        print(f"Mean Score:        {tn_example['mean_score']:.2f}")
        print(f"Original Tox Frac: {tn_example['tox_frac']:.3f} (Threshold: {config.USER_TOX_THRESHOLD})")
        print(f"Predicted Prob:    {tn_example['predicted_proba']:.4f} -> Label: {tn_example['predicted_label']}")
        print(f"True Weak Label:   {tn_example['weak_label']}")
        # Show the single most toxic post for the TN user
        display_top_posts(df_posts_raw, tn_example['username'], num_posts=1, title="Single Most Toxic Post")
    else:
        print("\n--- True Negative ---")
        print("  (No example found in test set)")

    # --- False Positive ---
    if fp_example is not None:
        print("\n--- False Positive (Predicted Toxic [✗], Actually Non-Toxic [✓]) ---")
        print(f"Username:          {fp_example['username']}")
        print(f"N Posts:           {fp_example['n_posts']}")
        print(f"Mean Score:        {fp_example['mean_score']:.2f}")
        print(f"Original Tox Frac: {fp_example['tox_frac']:.3f} (Threshold: {config.USER_TOX_THRESHOLD})")
        print(f"Predicted Prob:    {fp_example['predicted_proba']:.4f} -> Label: {fp_example['predicted_label']}")
        print(f"True Weak Label:   {fp_example['weak_label']}")
        # Show top posts to see why model might have predicted toxic
        display_top_posts(df_posts_raw, fp_example['username'], num_posts=2, title="Top 2 Most Toxic Posts")
    else:
        print("\n--- False Positive ---")
        print("  (No example found in test set)")

    # --- False Negative ---
    if fn_example is not None:
        print("\n--- False Negative (Predicted Non-Toxic [✗], Actually Toxic [✓]) ---")
        print(f"Username:          {fn_example['username']}")
        print(f"N Posts:           {fn_example['n_posts']}")
        print(f"Mean Score:        {fn_example['mean_score']:.2f}")
        print(f"Original Tox Frac: {fn_example['tox_frac']:.3f} (Threshold: {config.USER_TOX_THRESHOLD})")
        print(f"Predicted Prob:    {fn_example['predicted_proba']:.4f} -> Label: {fn_example['predicted_label']}")
        print(f"True Weak Label:   {fn_example['weak_label']}")
        # Show top posts to see why user was labeled toxic but model missed it
        display_top_posts(df_posts_raw, fn_example['username'], num_posts=2, title="Top 2 Most Toxic Posts")
    else:
        print("\n--- False Negative ---")
        print("  (No example found in test set)")


    # 6. Identify and Display Top N Predicted Toxic Users
    print("\n" + "="*60)
    print(f" TOP {TOP_N_USERS} USERS PREDICTED AS TOXIC (Ranked by Highest Probability)")
    print("="*60)

    top_toxic_predictions = df_test.sort_values('predicted_proba', ascending=False).head(TOP_N_USERS)

    if top_toxic_predictions.empty:
        print("No users predicted as toxic found in the test set.")
    else:
        # Use enumerate starting from 1 for correct rank
        for rank, (index, user_data) in enumerate(top_toxic_predictions.iterrows(), start=1):
            print(f"\n--- Rank {rank} Predicted Toxic User ---")
            print(f"Username:          {user_data['username']}")
            print(f"N Posts:           {user_data['n_posts']}")
            print(f"Mean Score:        {user_data['mean_score']:.2f}")
            print(f"Original Tox Frac: {user_data['tox_frac']:.3f} (Threshold: {config.USER_TOX_THRESHOLD})")
            print(f"Predicted Prob:    {user_data['predicted_proba']:.4f} -> Label: {user_data['predicted_label']}")
            # Check if the prediction matches the true label for these top predictions
            is_correct = user_data['weak_label'] == user_data['predicted_label']
            print(f"True Weak Label:   {user_data['weak_label']} {'(Correct)' if is_correct else '(Misclassified!)'}")

            # Fetch and display the single most toxic post for this user
            display_top_posts(df_posts_raw, user_data['username'], num_posts=1, title="Single Most Toxic Post")

    print("\n" + "="*60)
    print("Script finished.")

# --- Run the main function ---
if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please ensure the required model, scaler, test data, raw posts, and user labels files exist in the paths specified in config.py.")
    except ValueError as e:
        print(f"\nError: {e}")
        print("Please check the data format and column names in the input files.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

