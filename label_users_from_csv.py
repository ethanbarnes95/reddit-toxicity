# label_users_from_csv.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import config # Assuming config.py defines USER_TOX_THRESHOLD, RAW_POSTS_CSV, USER_LABELS_CSV

# Define output directory for plots
PLOT_OUTPUT_DIR = "eda_plots_labeling" # Or choose another directory name
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

# --- 1. Load Raw Post Data ---
print(f"Loading raw posts from {config.RAW_POSTS_CSV}...")
try:
    # Specify dtypes for potentially problematic columns if needed
    # dtype_spec = {'column_name': str} # Example
    df = pd.read_csv(config.RAW_POSTS_CSV) # Add low_memory=False if dtype warnings appear
except FileNotFoundError:
    print(f"Error: Raw posts file not found at {config.RAW_POSTS_CSV}")
    exit()
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()

print(f"Loaded {len(df)} rows.")

# --- 2. Data Cleaning (Optional but Recommended) ---
# Optional: Remove exact duplicate posts if you suspect collection issues
initial_rows = len(df)
df = df.drop_duplicates(subset="id")
if len(df) < initial_rows:
    print(f"Removed {initial_rows - len(df)} duplicate post entries based on 'id'.")

# Drop rows where username might be missing (if applicable)
df.dropna(subset=['username'], inplace=True)
print(f"Processing {len(df)} unique posts with valid usernames.")

# --- 3. Flag Toxic Posts ---
# Ensure 'tox_score' column exists
if "tox_score" not in df.columns:
    print("Error: 'tox_score' column not found in the CSV.")
    exit()

# Handle potential non-numeric values in tox_score before comparison
df['tox_score'] = pd.to_numeric(df['tox_score'], errors='coerce')
df.dropna(subset=['tox_score'], inplace=True) # Remove rows where tox_score couldn't be converted

print(f"Using post toxicity threshold (TOX_THRESHOLD): {config.TOX_THRESHOLD}")
df["is_tox_post"] = df["tox_score"] >= config.TOX_THRESHOLD

# --- 4. Aggregate Per User ---
print("Aggregating post toxicity per user...")
# Ensure 'is_tox_post' column exists after potential drops
if "is_tox_post" not in df.columns:
     print("Error: 'is_tox_post' column missing before aggregation.")
     exit()

g = df.groupby("username")
# Use agg method for clarity and potential future aggregations
agg = g.agg(
    tox_posts=('is_tox_post', 'sum'),
    posts=('is_tox_post', 'count')
).reset_index()

# --- 5. Calculate Toxicity Fraction and Weak Label ---
# Avoid division by zero if a user somehow has 0 posts after filtering
agg = agg[agg["posts"] > 0]

agg["tox_frac"] = agg["tox_posts"] / agg["posts"]

# Apply the initial user threshold for the weak label (needed for training)
print(f"Using initial user labeling threshold (USER_TOX_THRESHOLD): {config.USER_TOX_THRESHOLD}")
agg["weak_label"] = (agg["tox_frac"] >= config.USER_TOX_THRESHOLD).astype(int)

# --- 6. Analyze and Plot tox_frac Distribution ---
print("\n--- Analyzing User Toxicity Fraction ---")
print(agg["tox_frac"].describe())

plt.figure(figsize=(12, 6))
sns.histplot(agg["tox_frac"], bins=50, kde=False) # Use more bins for finer detail
plt.title('Distribution of User Toxicity Fraction (`tox_frac`)')
plt.xlabel('Fraction of Toxic Posts per User (`tox_frac`)')
plt.ylabel('Number of Users')
plt.axvline(config.USER_TOX_THRESHOLD, color='r', linestyle='--',
            label=f'Initial Threshold = {config.USER_TOX_THRESHOLD:.2f}')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the plot
plot_path = os.path.join(PLOT_OUTPUT_DIR, "user_tox_frac_distribution.png")
plt.savefig(plot_path)
print(f"Saved toxicity fraction distribution plot to: {plot_path}")
plt.close() # Close the plot to free memory

# --- 7. Save User Labels ---
# Select and rename columns for the final output CSV
output_df = agg[["username", "posts", "tox_posts", "tox_frac", "weak_label"]]

try:
    output_df.to_csv(config.USER_LABELS_CSV, index=False)
    print(f"\n✓ Wrote {len(output_df)} user rows to {config.USER_LABELS_CSV}")
    # Display class balance based on the initial threshold
    print("\nClass distribution based on initial threshold:")
    print(output_df["weak_label"].value_counts(normalize=True))
except Exception as e:
    print(f"Error saving user labels CSV: {e}")

