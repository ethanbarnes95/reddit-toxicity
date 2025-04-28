# eda_user_level_data_v2.py
"""
Performs Exploratory Data Analysis (EDA) on the processed user-level data.
Generates statistics and visualizations for the project report.

Reads: config.USER_PARQ (data/processed/user_level.parquet)
Outputs: Prints statistics and saves plots to an 'eda_plots' directory.

Changes:
- Plots histograms in separate figures for better readability.
- Adds optional x-axis limiting for mean_score histogram based on quantiles.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
# Assuming 'config.py' exists in the same directory or Python path
try:
    import config
    # Define default values if not found in config, or handle missing config more robustly
    USER_PARQ = getattr(config, 'USER_PARQ', 'data/processed/user_level.parquet')
    EMBEDDING_DIM = getattr(config, 'EMBEDDING_DIM', 768) # Example default dimension
    RANDOM_SEED = getattr(config, 'RANDOM_SEED', 42)
except ImportError:
    print("Warning: config.py not found. Using default paths and values.")
    # Define default paths and parameters if config is missing
    USER_PARQ = 'data/processed/user_level.parquet'
    EMBEDDING_DIM = 768 # Provide a sensible default
    RANDOM_SEED = 42
    # Create dummy data directory if it doesn't exist for the script to run
    os.makedirs('data/processed', exist_ok=True)
    # Consider creating a dummy parquet file for testing if needed, or exit
    # For now, we'll let it fail later if the file is truly missing

# --- Configuration ---
USER_DATA_PATH = USER_PARQ
PLOT_OUTPUT_DIR = "eda_plots"
# Reduce sample size for t-SNE if dataset is very large or for faster plotting
TSNE_SAMPLE_SIZE = None # Set to an integer like 2000 to sample, None for full dataset
TSNE_PERPLEXITY = 30 # Typical value, adjust based on dataset size/density
# Quantiles for limiting mean_score x-axis (e.g., 0.01 and 0.99 exclude 1% tails)
MEAN_SCORE_XLIM_QUANTILES = [0.01, 0.99]

# --- Create output directory ---
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

# --- Load Data ---
print(f"Loading user data from {USER_DATA_PATH}...")
try:
    df = pd.read_parquet(USER_DATA_PATH)
    print(f"Loaded {len(df)} user records.")
except Exception as e:
    print(f"Error loading data from {USER_DATA_PATH}: {e}")
    # Create a dummy DataFrame for demonstration if loading fails
    print("Creating dummy data for demonstration purposes.")
    data = {
        'user_id': [f'user_{i}' for i in range(2000)],
        'n_posts': np.random.lognormal(mean=3, sigma=1.5, size=2000).astype(int) + 1,
        # Skewed mean_score with some outliers
        'mean_score': np.concatenate([
            np.random.normal(loc=10, scale=20, size=1800), # Main cluster
            np.random.normal(loc=500, scale=100, size=150), # Some higher scores
            np.random.normal(loc=-50, scale=10, size=50) # Some negative scores
        ]),
        'weak_label': np.random.randint(0, 2, size=2000),
        # Dummy embeddings (replace with actual logic if needed)
        'emb': [np.random.rand(EMBEDDING_DIM).tolist() for _ in range(2000)]
    }
    df = pd.DataFrame(data)
    # Ensure correct types for dummy data
    df['n_posts'] = pd.to_numeric(df['n_posts'], errors='coerce')
    df['mean_score'] = pd.to_numeric(df['mean_score'], errors='coerce')
    df['weak_label'] = df['weak_label'].astype(int)
    df.dropna(subset=['n_posts', 'mean_score'], inplace=True)
    print(f"Using dummy data with {len(df)} records.")


# --- Basic Info & Class Distribution ---
print("\n--- Basic Data Info ---")
# Use try-except for info() in case df is None after failed load and no dummy data
try:
    df.info()
except AttributeError:
    print("DataFrame not loaded. Cannot display info.")
    exit() # Exit if no data

print("\n--- Class Distribution ---")
if 'weak_label' in df.columns:
    class_counts = df['weak_label'].value_counts()
    class_perc = df['weak_label'].value_counts(normalize=True) * 100
    print(class_counts)
    print(f"\nNon-toxic (0): {class_perc.get(0, 0):.2f}%")
    print(f"Toxic (1):     {class_perc.get(1, 0):.2f}%")
else:
    print("Column 'weak_label' not found.")


# --- Statistical Summaries ---
print("\n--- Statistical Summaries (Grouped by Class) ---")
# Ensure numeric columns exist and are numeric
numeric_cols = ['n_posts', 'mean_score']
valid_cols = [col for col in numeric_cols if col in df.columns]

if not valid_cols:
    print("Required numeric columns ('n_posts', 'mean_score') not found.")
else:
    for col in valid_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=valid_cols, inplace=True)

    if 'weak_label' in df.columns and valid_cols:
        summary_stats = df.groupby('weak_label')[valid_cols].agg(
            ['mean', 'median', 'std', 'min', 'max', 'count']
        )
        print(summary_stats)
    elif not valid_cols:
         print("No valid numeric columns to summarize.")
    else:
        print("Column 'weak_label' not found for grouping.")


# --- Visualizations ---
print("\nGenerating plots...")
sns.set_theme(style="whitegrid")

# 1a. Histogram for n_posts
if 'n_posts' in df.columns and 'weak_label' in df.columns:
    print("Generating n_posts histogram...")
    plt.figure(figsize=(8, 6)) # Adjusted figure size for single plot
    sns.histplot(data=df, x='n_posts', hue='weak_label', kde=True, log_scale=True, palette='viridis')
    plt.title('Distribution of n_posts (Log Scale)')
    plt.xlabel('Number of Posts (Log Scale)')
    plt.ylabel('Number of Users')
    plt.tight_layout()
    hist_path_nposts = os.path.join(PLOT_OUTPUT_DIR, "histogram_nposts.png")
    try:
        plt.savefig(hist_path_nposts)
        print(f"Saved n_posts histogram to {hist_path_nposts}")
    except Exception as e:
        print(f"Error saving n_posts histogram: {e}")
    plt.close() # Close the figure
else:
    print("Skipping n_posts histogram: 'n_posts' or 'weak_label' column missing.")

# 1b. Histogram for mean_score
if 'mean_score' in df.columns and 'weak_label' in df.columns:
    print("Generating mean_score histogram...")
    plt.figure(figsize=(8, 6)) # Adjusted figure size for single plot
    sns.histplot(data=df, x='mean_score', hue='weak_label', kde=True, palette='viridis')

    # --- Optional: Limit x-axis based on quantiles ---
    try:
        # Calculate quantiles only on valid, finite data
        valid_scores = df.loc[np.isfinite(df['mean_score']), 'mean_score']
        if not valid_scores.empty:
            low_lim, high_lim = valid_scores.quantile(MEAN_SCORE_XLIM_QUANTILES)
            # Add a small buffer if limits are too close, or if one limit is extreme
            buffer = (high_lim - low_lim) * 0.05 # 5% buffer
            low_lim -= buffer
            high_lim += buffer
            print(f"Setting mean_score x-axis limits based on quantiles {MEAN_SCORE_XLIM_QUANTILES}: ({low_lim:.2f}, {high_lim:.2f})")
            plt.xlim(low_lim, high_lim)
        else:
            print("No valid finite 'mean_score' values to calculate quantiles.")
    except Exception as e:
        print(f"Could not calculate or apply x-axis limits for mean_score: {e}")
    # --- End Optional ---

    plt.title('Distribution of mean_score')
    plt.xlabel('Mean Post Score')
    plt.ylabel('Number of Users')
    plt.tight_layout()
    hist_path_meanscore = os.path.join(PLOT_OUTPUT_DIR, "histogram_meanscore.png")
    try:
        plt.savefig(hist_path_meanscore)
        print(f"Saved mean_score histogram to {hist_path_meanscore}")
    except Exception as e:
        print(f"Error saving mean_score histogram: {e}")
    plt.close() # Close the figure
else:
    print("Skipping mean_score histogram: 'mean_score' or 'weak_label' column missing.")


# 2. Scatter plot of mean_score vs n_posts
if all(col in df.columns for col in ['n_posts', 'mean_score', 'weak_label']):
    print("Generating scatter plot...")
    plt.figure(figsize=(8, 6))
    # Filter out non-finite values before plotting
    plot_data = df[['n_posts', 'mean_score', 'weak_label']].copy()
    plot_data = plot_data.replace([np.inf, -np.inf], np.nan).dropna()

    # Ensure n_posts > 0 for log scale
    plot_data = plot_data[plot_data['n_posts'] > 0]

    if not plot_data.empty:
        sns.scatterplot(data=plot_data, x='n_posts', y='mean_score', hue='weak_label', alpha=0.6, palette='viridis', s=50)
        plt.xscale('log') # Use log scale for n_posts due to potential skew
        plt.title('mean_score vs n_posts (Log Scale)')
        plt.xlabel('Number of Posts (Log Scale)')
        plt.ylabel('Mean Post Score')
        plt.tight_layout()
        scatter_path = os.path.join(PLOT_OUTPUT_DIR, "scatter_meanscore_vs_nposts.png")
        try:
            plt.savefig(scatter_path)
            print(f"Saved scatter plot to {scatter_path}")
        except Exception as e:
            print(f"Error saving scatter plot: {e}")
    else:
        print("No valid data points for scatter plot after filtering.")
    plt.close() # Close the figure
else:
    print("Skipping scatter plot: One or more required columns ('n_posts', 'mean_score', 'weak_label') missing.")


# 3. t-SNE Visualization of User Embeddings
print("\nPreparing t-SNE plot (this may take a while)...")
# Check if 'emb' column exists and is usable
if 'emb' not in df.columns or 'weak_label' not in df.columns:
    print("Error: 'emb' or 'weak_label' column not found in the dataframe. Skipping t-SNE.")
else:
    try:
        # Filter out rows where 'emb' is None or NaN before processing
        valid_emb_df = df[['emb', 'weak_label']].dropna(subset=['emb'])

        # Convert list/array embeddings to a NumPy array
        embeddings_list = valid_emb_df['emb'].tolist()
        labels_list = valid_emb_df['weak_label'].values # Get corresponding labels

        if not embeddings_list:
            print("No valid embeddings found to plot.")
        else:
            # Robustly handle potential variations in list content
            try:
                embeddings = np.array(embeddings_list, dtype=np.float32)
                # Check if conversion resulted in an object array (e.g., due to inconsistent lengths)
                if embeddings.dtype == 'object':
                     raise ValueError("Embeddings have inconsistent shapes or types.")
            except ValueError as ve:
                 print(f"Error converting embeddings to NumPy array: {ve}")
                 # Attempt to filter out problematic embeddings if possible, or skip
                 # Example: Filter based on expected dimension (if known)
                 try:
                     embeddings = np.array([e for e in embeddings_list if len(e) == EMBEDDING_DIM], dtype=np.float32)
                     # Get corresponding labels again after filtering
                     labels_list = valid_emb_df.loc[[len(e) == EMBEDDING_DIM for e in embeddings_list], 'weak_label'].values
                     if embeddings.size == 0:
                         print("No embeddings with expected dimension found.")
                         raise ValueError("No valid embeddings after filtering.")
                     print(f"Warning: Filtered embeddings based on expected dimension ({EMBEDDING_DIM}).")
                 except Exception as filter_e:
                     print(f"Could not filter embeddings. Skipping t-SNE. Error: {filter_e}")
                     embeddings = None # Ensure we skip the rest

            if embeddings is not None and embeddings.ndim == 2: # Proceed only if we have a valid 2D array
                labels = np.array(labels_list)

                # Check embedding dimension consistency (optional but recommended)
                if embeddings.shape[1] != EMBEDDING_DIM:
                    print(f"Warning: Embeddings have unexpected dimension {embeddings.shape[1]}. Expected {EMBEDDING_DIM}.")
                    # Proceeding anyway, but check aggregation script if this occurs

                # Optional sampling for large datasets
                if TSNE_SAMPLE_SIZE and TSNE_SAMPLE_SIZE < len(embeddings):
                    print(f"Sampling {TSNE_SAMPLE_SIZE} points for t-SNE...")
                    indices = np.random.choice(len(embeddings), TSNE_SAMPLE_SIZE, replace=False)
                    embeddings_sample = embeddings[indices]
                    labels_sample = labels[indices]
                else:
                    embeddings_sample = embeddings
                    labels_sample = labels

                # Perform t-SNE
                print(f"Running t-SNE on {len(embeddings_sample)} samples...")
                tsne = TSNE(n_components=2, random_state=RANDOM_SEED,
                            perplexity=TSNE_PERPLEXITY, n_iter=300, n_jobs=-1, init='pca', learning_rate='auto') # Added init/lr
                embeddings_2d = tsne.fit_transform(embeddings_sample)
                print("t-SNE finished.")

                # Create plot
                plt.figure(figsize=(10, 8))
                plot_df_tsne = pd.DataFrame({'tsne_1': embeddings_2d[:, 0], 'tsne_2': embeddings_2d[:, 1], 'label': labels_sample})
                sns.scatterplot(data=plot_df_tsne, x='tsne_1', y='tsne_2', hue='label',
                                palette='viridis', alpha=0.7, s=50)

                plt.title(f't-SNE Visualization of User Embeddings (Perplexity={TSNE_PERPLEXITY})')
                plt.xlabel('t-SNE Component 1')
                plt.ylabel('t-SNE Component 2')
                handles, _ = plt.gca().get_legend_handles_labels()
                # Ensure correct legend labels even if sampling resulted in only one class
                legend_labels = []
                unique_labels = np.unique(labels_sample)
                if 0 in unique_labels: legend_labels.append('Non-toxic (0)')
                if 1 in unique_labels: legend_labels.append('Toxic (1)')
                # Ensure handles match labels found
                if len(handles) == len(legend_labels):
                    plt.legend(handles, legend_labels, title='Class')
                else: # Fallback if something is odd (e.g., only one class plotted)
                    plt.legend(title='Class')


                tsne_path = os.path.join(PLOT_OUTPUT_DIR, "tsne_user_embeddings.png")
                try:
                    plt.savefig(tsne_path)
                    print(f"Saved t-SNE plot to {tsne_path}")
                except Exception as e:
                    print(f"Error saving t-SNE plot: {e}")
                plt.close() # Close the figure
            elif embeddings is not None:
                 print(f"Skipping t-SNE: Embeddings array has unexpected shape {embeddings.shape} after processing.")

    except Exception as e:
        print(f"Error during t-SNE processing or plotting: {e}")
        # Ensure plot is closed if error occurred mid-plot
        plt.close()


print("\nEDA script finished.")
