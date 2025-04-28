"""
Build per‑user feature table with *no* direct toxicity features.

Reads:
  • config.RAW_POSTS_CSV
  • config.POST_EMB_PARQ
  • config.USER_LABELS_CSV
Writes:
  • config.USER_PARQ
"""

import numpy as np
import polars as pl
import config
from pathlib import Path
import sys # For exit

# Ensure output directory exists
Path(config.PROC_DIR).mkdir(parents=True, exist_ok=True)
EXPECTED_DIM = config.EMBEDDING_DIM

# ---------- Helper: Check if files exist ----------
def check_file_exists(filepath):
    if not Path(filepath).is_file():
        print(f"Error: Input file not found at {filepath}")
        sys.exit(1) # Exit if critical input is missing

# ---------- 1: Read posts, embeddings, and labels ----------
print("Checking input files...")
check_file_exists(config.RAW_POSTS_CSV)
check_file_exists(config.POST_EMB_PARQ)
check_file_exists(config.USER_LABELS_CSV)

print("Loading data...")
try:
    # Select only necessary columns from raw posts to save memory
    posts = pl.read_csv(config.RAW_POSTS_CSV,
                        columns=['username', 'id', 'score'], # Only load needed columns
                        low_memory=True, null_values=["", "None", "nan"])
    embs  = pl.read_parquet(config.POST_EMB_PARQ) # Parquet is efficient
    labels = pl.read_csv(config.USER_LABELS_CSV,
                         columns=['username', 'weak_label']) # Only load needed columns
except Exception as e:
    print(f"Error loading input files: {e}")
    sys.exit(1)

# Drop rows with null usernames or scores if they exist after loading
posts = posts.drop_nulls(subset=['username', 'score'])
# Convert score to float for aggregation if it's not already
posts = posts.with_columns(pl.col('score').cast(pl.Float64, strict=False))


print(f"Loaded {posts.height} post entries.")
print(f"Loaded {embs.height} embeddings.")
print(f"Loaded {labels.height} user labels.")

# ---------- 2: Join posts and embeddings ----------
print("Joining posts and embeddings...")
# Use inner join to keep only posts that have embeddings
df = posts.join(embs, on="id", how="inner")
print(f"Joined data has {df.height} rows.")

# Check if join resulted in empty dataframe
if df.height == 0:
    print("Error: No matching 'id' found between posts and embeddings. Check the 'id' columns.")
    sys.exit(1)

# ---------- 3: Aggregate per user (no tox_ columns) ----------
print("Aggregating features per user...")

# Define aggregation logic using Polars expressions for potential speedup
# compared to map_groups for simple aggregations.
# For embedding mean, map_groups or a custom function might still be needed
# if direct Polars aggregation isn't straightforward.

# Let's stick with map_groups for consistency with the original approach,
# as it handles the numpy embedding aggregation cleanly.
def agg_grp(grp: pl.DataFrame) -> pl.DataFrame:
    # Check if embedding column is present and not empty
    if "emb" not in grp.columns or grp["emb"].is_null().all():
         # Handle cases where a user group might lack valid embeddings
         # Return a DataFrame with nulls or default values for this group
         # Or filter such groups out beforehand if appropriate
         print(f"Warning: User group {grp['username'][0]} has missing/null embeddings. Skipping mean embedding.")
         # Option 1: Return null embedding
         # mean_emb = [None] * 384 # Assuming 384 dimensions
         # Option 2: Skip user (will be filtered out later by inner join with labels potentially)
         return None # Or return an empty DataFrame: pl.DataFrame()

    # Convert embedding list/series to numpy array
    try:
        # Assuming 'emb' column contains lists/arrays
        mat = np.array(grp["emb"].to_list(), dtype=np.float32)
        if mat.ndim != 2 or mat.shape[1] != EXPECTED_DIM:  # Basic check on embedding shape
            raise ValueError(
                f"Incorrect embedding shape {mat.shape} for user {grp['username'][0]}. Expected dim {EXPECTED_DIM}")
        mean_emb = mat.mean(axis=0).tolist()
    except Exception as e:
        print(f"Error processing embeddings for user {grp['username'][0]}: {e}")
        # Decide how to handle: skip user, return null embedding, etc.
        mean_emb = [np.nan] * EXPECTED_DIM # Return NaNs for embedding if error occurs


    # Calculate other aggregates
    # Ensure 'score' is numeric before calculating mean
    mean_score_val = grp["score"].mean()

    return pl.DataFrame({
        "username":   [grp["username"][0]],
        "n_posts":    [len(grp)], # Count of posts *with embeddings* for this user
        "mean_score": [mean_score_val],
        # Add other features if needed: std_score, median_score, etc.
        # "std_score": [grp["score"].std()],
        "emb":        [mean_emb], # Mean embedding vector
    })

# Apply the aggregation
# Note: map_groups can be slower than pure Polars aggregations for very large datasets
# but is flexible for complex operations like the embedding mean.
user_features = (
    df
    .group_by("username")
    .map_groups(agg_grp)
)

# Filter out any potential null results from agg_grp if users were skipped
user_features = user_features.drop_nulls()

print(f"Aggregated features for {user_features.height} users.")

# ---------- 4: Attach weak labels ----------
print("Attaching weak labels...")
# Use inner join: only keep users who have both features and a label
# This implicitly filters out users who might have been in raw_posts but not in user_labels
user_df_final = user_features.join(
    labels.select(["username", "weak_label"]), # Select only needed columns from labels
    on="username",
    how="inner"
)

print(f"Final dataset contains {user_df_final.height} users with features and labels.")

# Check if the final DataFrame is empty
if user_df_final.height == 0:
    print("Error: No users remaining after joining features and labels. Check 'username' matching.")
    sys.exit(1)


# ---------- 5: Save ----------
print(f"Saving user-level feature table to {config.USER_PARQ}...")
try:
    user_df_final.write_parquet(config.USER_PARQ)

    # Display final class counts
    # Use value_counts directly on the Polars DataFrame column
    counts_df = user_df_final.get_column("weak_label").value_counts()
    print(f"✓ User-level table saved successfully.")
    print("\nFinal Class Counts in Parquet:")
    print(counts_df)

    # Optional: Print proportions
    total_users = user_df_final.height
    proportions = counts_df.with_columns(
        (pl.col("counts") / total_users).alias("proportion")
    )
    print("\nFinal Class Proportions:")
    print(proportions)

except Exception as e:
    print(f"Error writing final Parquet file {config.USER_PARQ}: {e}")

