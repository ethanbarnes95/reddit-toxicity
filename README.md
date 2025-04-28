# Reddit User Toxicity Prediction Project

This project aims to identify potentially toxic users on Reddit based on their posting history. It involves collecting Reddit posts and comments, processing the text data, calculating toxicity scores, generating user-level features (including text embeddings), training machine learning models (baseline and XGBoost), and evaluating their performance.

## Project Structure

The project follows a typical machine learning pipeline:

1.  **Data Collection:** Scrapes posts and comments for users from specified subreddits using the Reddit API (PRAW).
2.  **Data Labeling:** Calculates post-level toxicity scores using a pre-trained transformer model (`unitary/toxic-bert`) and assigns a weak label (toxic/non-toxic) to users based on the fraction of their posts exceeding a toxicity threshold.
3.  **Feature Preprocessing:** Cleans post text (lowercase, removes URLs, HTML, emojis) and generates sentence embeddings for each post using a Sentence-BERT model (e.g., `all-MiniLM-L6-v2`).
4.  **Feature Aggregation:** Aggregates post-level information (embeddings, scores) to the user level, creating features like the mean post score, number of posts, and mean user embedding.
5.  **Dataset Splitting:** Splits the user-level dataset into training, validation, and test sets using stratified sampling to maintain class balance.
6.  **EDA:** Performs Exploratory Data Analysis on the aggregated user-level data to understand distributions and relationships.
7.  **Model Training:**
    * Trains baseline models (Logistic Regression, Random Forest) using only numeric features.
    * Trains an XGBoost classifier using both numeric features and user embeddings. This involves scaling numeric features, applying SMOTETomek for handling class imbalance, tuning hyperparameters with Optuna, and evaluating performance.
8.  **Model Evaluation & Analysis:** Evaluates the final model using metrics like Average Precision (AP), classification reports, confusion matrices, and Precision-Recall curves. Also analyzes feature importance and generates learning curves.
9.  **Prediction Examples:** Provides functionality to load the trained model and display specific examples of correct and incorrect predictions (True Positives, True Negatives, False Positives, False Negatives) along with the user's most toxic posts.

## Configuration (`config.py`)

This file centralizes important parameters:

* **Subreddits:** Lists for positive (potentially toxic) and control subreddits.
* **Scraping Limits:** Defines limits for data collection (time filter, posts per sub, posts/comments per user, minimum activity, account age).
* **Models:** Specifies the Sentence-BERT model (`SBERT_MODEL_NAME`) and the toxicity classification model (`TOX_MODEL_NAME`).
* **Thresholds:** Sets the post toxicity threshold (`TOX_THRESHOLD`) and the user-level weak labeling threshold (`USER_TOX_THRESHOLD`).
* **Features:** Defines numeric columns (`NUMERIC_COLS`) and embedding dimension (`EMBEDDING_DIM`).
* **Paths:** Specifies directories and filenames for raw data, processed data, splits, and models.
* **Misc:** Random seed for reproducibility.

## File Descriptions

* **`config.py`**: Main configuration file.
* **`data_collection.py`**: Collects posts/comments, scores toxicity, and saves raw posts (`posts_raw.csv`) and initial user labels (`user_labels.csv`).
* **`preprocess_features.py`**: Cleans text and generates post embeddings (`post_emb.parquet`) from `posts_raw.csv`.
* **`label_users_from_csv.py`**: Calculates user toxicity fraction and weak labels from `posts_raw.csv`, saving to `user_labels.csv`. Includes plotting the distribution of the toxicity fraction. *(Note: `data_collection.py` also generates a `user_labels.csv`)*.
* **`aggregate_users.py`**: Combines raw post scores, post embeddings, and user labels to create the final user-level feature set (`user_level.parquet`).
* **`split_dataset.py`**: Splits `user_level.parquet` into stratified train, validation, and test sets (`train.parquet`, `valid.parquet`, `test.parquet`).
* **`eda_user_level_data_v2.py`**: Performs EDA on `user_level.parquet` and saves plots.
* **`train_baseline.py`**: Trains and evaluates Logistic Regression and Random Forest models using only numeric features from the split data.
* **`train_xgboost_smotetomek_fixed.py`**: Trains, tunes (Optuna), and evaluates the main XGBoost model using combined numeric and embedding features, applying SMOTETomek. Saves the final model and scaler.
* **`show_model_examples.py`**: Loads the trained XGBoost model and test data to display illustrative prediction examples (TP, TN, FP, FN).

## Setup & Usage

1.  **Environment:**
    * Install required Python packages (e.g., `pandas`, `praw`, `prawcore`, `transformers`, `torch`, `polars`, `scikit-learn`, `xgboost`, `optuna`, `matplotlib`, `seaborn`, `imblearn`, `python-dotenv`, `tqdm`).
    * Set up Reddit API credentials in a `.env` file (see `data_collection.py` for required variables: `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT`, `REDDIT_USERNAME`, `REDDIT_PASSWORD`).
2.  **Configuration:** Adjust parameters in `config.py` as needed (subreddit lists, paths, models, thresholds).
3.  **Run Pipeline:** Execute the scripts in the following order:
    * `python data_collection.py`
    * `python preprocess_features.py`
    * *(Optional/Alternative Labeling)* `python label_users_from_csv.py`
    * `python aggregate_users.py`
    * `python split_dataset.py`
    * *(Optional EDA)* `python eda_user_level_data_v2.py`
    * *(Optional Baseline)* `python train_baseline.py`
    * `python train_xgboost_smotetomek_fixed.py`
    * *(Optional Examples)* `python show_model_examples.py`
4.  **Outputs:**
    * Raw data and labels in `data/raw/`.
    * Processed data (embeddings, user features, splits) in `data/processed/`.
    * Trained models and scalers in `models/`.
    * EDA and evaluation plots in specified directories (e.g., `eda_plots/`, `eda_plots_labeling/`, `eda_plots_fixed/`).

## Key Dependencies

* **PRAW:** For interacting with the Reddit API.
* **Transformers (Hugging Face):** For loading pre-trained toxicity and embedding models.
* **PyTorch:** As the backend for the transformer models.
* **Pandas / Polars:** For data manipulation.
* **Scikit-learn:** For data splitting, scaling, evaluation metrics, baseline models.
* **Imbalanced-learn:** For SMOTETomek sampling.
* **XGBoost:** For the gradient boosting model.
* **Optuna:** For hyperparameter optimization.
* **Matplotlib / Seaborn:** For plotting.
