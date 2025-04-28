# ------------------ subreddit design ------------------
POSITIVE_SUBS = ["4chan", "politics", "conspiracy", "Drama", "mensrights", "insanepeoplefacebook", "iamatotalpieceofshit"]
CONTROL_SUBS  = ["aww", "AskHistorians", "science", "gardening"]

# ------------------ scraping + labeling ----------------
TIME_FILTER            = "month"
SUB_POST_LIMIT         = 400
USER_POST_LIMIT        = 600
USER_COMMENT_LIMIT     = 600
MIN_ACTIVITY           = 100
MIN_ACCOUNT_AGE_DAYS   = 180
MAX_RETRIES_PER_USER   = 3

# ------------------ Feature Engineering ---------------
SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384 # 768 for mpnet-base-v2, 384 for MiniLM-L6-v2
#SBERT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2" # Or "sentence-transformers/all-MiniLM-L6-v2"
#EMBEDDING_DIM = 768 # 768 for mpnet-base-v2, 384 for MiniLM-L6-v2

TOX_MODEL_NAME     = "unitary/toxic-bert"
TOX_THRESHOLD      = 0.60
USER_TOX_THRESHOLD = 0.15          # mark user toxic if ≥ 25 % toxic posts

NUMERIC_COLS = ["n_posts", "mean_score"]

# ------------------ paths ------------------------------
RAW_DIR         = "data/raw"
PROC_DIR        = "data/processed"
RAW_POSTS_CSV   = f"{RAW_DIR}/posts_raw.csv"
USER_LABELS_CSV = f"{RAW_DIR}/user_labels.csv"
POST_EMB_PARQ   = f"{PROC_DIR}/post_emb.parquet"
USER_PARQ       = f"{PROC_DIR}/user_level.parquet"
SPLIT_DIR       = f"{PROC_DIR}/splits"
MODEL_DIR       = "models"

# ------------------ misc -------------------------------
RANDOM_SEED = 42
