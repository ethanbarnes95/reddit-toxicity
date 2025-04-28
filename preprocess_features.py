# preprocess_features.py
"""
Reads data/posts_raw.csv
• clean text (lower‑case, strip URLs, HTML, emoji)
• embed with specified SBERT model (e.g., all‑MiniLM‑L6‑v2 or all-mpnet-base-v2)
Writes data/post_embeddings/post_emb.parquet (id, emb[DIM])
"""

import os, re, html, torch
import polars as pl
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm
import config # Import project configuration
import sys # For exit

# --- Configuration from config.py ---
RAW_CSV    = config.RAW_POSTS_CSV
OUT_DIR    = config.PROC_DIR
OUT_FILE   = config.POST_EMB_PARQ
# Use the model name defined in config.py
MODEL_NAME = getattr(config, "SBERT_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2") # Default if not in config
# Embedding dimension is determined by the model, but good to have in config for other scripts
EXPECTED_DIM = getattr(config, "EMBEDDING_DIM", 384) # Default if not in config

BATCH_SIZE = 128 # Consider moving to config.py if you want to tune it easily

# Ensure output directory exists
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- 1. cleaning ----------
URL_RE  = re.compile(r"https?://\S+")
EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "]+", flags=re.UNICODE)

def clean(text: str) -> str:
    text = html.unescape(text or "")
    text = URL_RE.sub(" ", text)
    text = EMOJI_RE.sub(" ", text)
    text = text.lower()
    # Optional: remove extra whitespace
    text = ' '.join(text.split())
    return text.strip()

# ---------- 2. embedding ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Loading embedding model: {MODEL_NAME}")

try:
    tok  = AutoTokenizer.from_pretrained(MODEL_NAME)
    bert = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
    # Get the actual embedding dimension from the loaded model config
    actual_embedding_dim = bert.config.hidden_size
    print(f"Model loaded successfully. Detected embedding dimension: {actual_embedding_dim}")
    # Optional: Check against config value if it exists
    if hasattr(config, "EMBEDDING_DIM") and config.EMBEDDING_DIM != actual_embedding_dim:
         print(f"Warning: Detected dimension ({actual_embedding_dim}) differs from config.py ({config.EMBEDDING_DIM}). Using detected dimension.")
         # Update EXPECTED_DIM to actual for consistency if needed later
         EXPECTED_DIM = actual_embedding_dim

except Exception as e:
    print(f"Error loading model {MODEL_NAME}: {e}")
    sys.exit(1) # Exit if model loading fails

@torch.inference_mode()
def embed(batch_text):
    # Tokenize the batch of text
    enc = tok(batch_text, padding=True, truncation=True,
              return_tensors="pt", max_length=512).to(device) # Use model's max sequence length if appropriate

    # *** FIX START ***
    # Pass the tokenized input through the model
    outputs = bert(**enc)

    # Get the last hidden state from the MODEL's output
    last_hidden = outputs.last_hidden_state
    # *** FIX END ***

    # Get the attention mask from the TOKENIZER's output (needed for pooling)
    attention_mask = enc.attention_mask

    # Perform mean pooling using the attention mask to ignore padding tokens
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    sum_embeddings = torch.sum(last_hidden * mask_expanded, 1)
    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9) # Avoid division by zero
    mean_pooled = sum_embeddings / sum_mask

    return mean_pooled.cpu().numpy()

# ---------- 3. stream through CSV ----------
print(f"Reading CSV: {RAW_CSV}")
try:
    stream = pl.read_csv(
        RAW_CSV,
        low_memory=True,
        infer_schema_length=1000,
        null_values=["", "None", "nan"],
    )
    print(f"CSV columns: {stream.columns}")
except Exception as e:
    print(f"Error reading CSV file {RAW_CSV}: {e}")
    sys.exit(1)

required_cols = ['id', 'title', 'body']
if not all(col in stream.columns for col in required_cols):
     print(f"Error: Missing one or more required columns in {RAW_CSV}. Expected: {required_cols}")
     print(f"Found columns: {stream.columns}")
     sys.exit(1)

# Process in chunks
chunks, ids, texts = [], [], []
total_rows = stream.height

print("Starting embedding process...")
# Use select().iter_rows() for potentially better performance with Polars
for row in tqdm(stream.select(['id', 'title', 'body']).iter_rows(named=True), desc="Embedding posts", total=total_rows):
    title_text = row.get("title") or ""
    body_text = row.get("body") or ""
    combined_text = title_text + " " + body_text

    ids.append(row["id"])
    texts.append(clean(combined_text))

    if len(texts) == BATCH_SIZE:
        embs = embed(texts)
        # Check embedding dimension consistency (optional sanity check)
        if embs.shape[1] != actual_embedding_dim:
             print(f"Warning: Embedding dimension mismatch! Expected {actual_embedding_dim}, got {embs.shape[1]}. Check model/pooling.")
        chunks.append(pl.DataFrame({"id": ids, "emb": embs.tolist()}))
        ids, texts = [], []

if texts: # Process remainder
    embs = embed(texts)
    if embs.shape[1] != actual_embedding_dim:
             print(f"Warning: Embedding dimension mismatch! Expected {actual_embedding_dim}, got {embs.shape[1]}. Check model/pooling.")
    chunks.append(pl.DataFrame({"id": ids, "emb": embs.tolist()}))

# ---------- 4. Combine and Save ----------
if not chunks:
    print("Warning: No embeddings were generated. Check CSV content and filtering.")
else:
    print(f"\nCombining {len(chunks)} chunks...")
    final_df = pl.concat(chunks)
    # Check the dimension of the first embedding list as a proxy
    first_emb_len = len(final_df['emb'][0]) if final_df.height > 0 and final_df['emb'][0] is not None else 'N/A'
    print(f"Generated {final_df.height} embeddings with dimension {first_emb_len}.")

    try:
        final_df.write_parquet(OUT_FILE)
        print(f"✓ Embeddings written to {OUT_FILE}")
    except Exception as e:
        print(f"Error writing Parquet file {OUT_FILE}: {e}")

# Optional: Clean up GPU memory
# del bert
# del tok
# if torch.cuda.is_available():
#     torch.cuda.empty_cache()
print("Embedding script finished.")
