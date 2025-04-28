# data_collection.py
"""
Collect Reddit data *and* emit two files:
  1. data/posts_raw.csv   – submissions/comments + meta + tox scores
  2. data/user_labels.csv – per‑user weak label (1 toxic, 0 clean)
"""

import os, time, random, torch
from collections import defaultdict

import pandas as pd
import praw
import prawcore
from prawcore.exceptions import TooManyRequests, Forbidden, NotFound
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv

import config
load_dotenv()

# ----------------------------- constants & back‑off
BACKOFF_BASE = 30          # seconds on first 429
BACKOFF_MAX  = 300         # 5‑minute cap
backoff_delay = BACKOFF_BASE

# ----------------------------- helper: score toxicity
@torch.inference_mode()
def toxicity_score(text: str, bundle):
    tok, mdl, device = bundle
    if not text.strip():
        return 0.0, [0.0] * 6
    enc = tok(text[:512], truncation=True, return_tensors="pt").to(device)
    probs = torch.sigmoid(mdl(**enc).logits[0]).tolist()  # 6‑label
    return max(probs), probs

# ----------------------------- init Reddit & model
def init_clients():
    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent=os.getenv("REDDIT_USER_AGENT"),
        username=os.getenv("REDDIT_USERNAME"),     # gives higher quota
        password=os.getenv("REDDIT_PASSWORD"),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok  = AutoTokenizer.from_pretrained(config.TOX_MODEL_NAME)
    mdl = AutoModelForSequenceClassification.from_pretrained(
        config.TOX_MODEL_NAME, torch_dtype=torch.float16
    ).to(device).eval()
    return reddit, (tok, mdl, device)

# ----------------------------- main
def main():
    reddit, hf_bundle = init_clients()
    os.makedirs("data", exist_ok=True)

    tox_counts, total_counts = defaultdict(int), defaultdict(int)
    post_rows, done_users = [], set()

    subs_to_scan = config.POSITIVE_SUBS + config.CONTROL_SUBS
    random.shuffle(subs_to_scan)

    for sub in subs_to_scan:
        print(f"⇢  scanning r/{sub}")
        for post in reddit.subreddit(sub).top(time_filter=config.TIME_FILTER,
                                              limit=config.SUB_POST_LIMIT):
            post.comments.replace_more(limit=0)
            for thing in [post] + list(post.comments):
                if not thing or not getattr(thing, "author", None):
                    continue
                u = thing.author.name
                if u in done_users:
                    continue

                # ------- light user filters -------
                try:
                    redditor = reddit.redditor(u)
                    created_ts = getattr(redditor, "created_utc", None)
                    if created_ts is None:
                        continue
                    acct_age = (pd.Timestamp.utcnow().timestamp() - created_ts) / 86400
                    if acct_age < config.MIN_ACCOUNT_AGE_DAYS:
                        continue
                    subs  = list(redditor.submissions.new(limit=config.USER_POST_LIMIT))
                    comms = list(redditor.comments.new(limit=config.USER_COMMENT_LIMIT))
                except (Forbidden, NotFound, AttributeError):
                    continue
                except TooManyRequests:
                    print("⚠ global 429 – sleeping 60 s")
                    time.sleep(60)
                    continue

                acts = subs + comms
                if len(acts) < config.MIN_ACTIVITY:
                    continue

                # ------- score each act -------
                global backoff_delay
                for act in acts:
                    # inner 429 guard around body/selftext fetch
                    try:
                        text = (getattr(act, "selftext", "") or
                                getattr(act, "body", "") or "")
                    except prawcore.exceptions.TooManyRequests:
                        print(f"⚠  429 inside act – sleeping {backoff_delay}s")
                        time.sleep(backoff_delay)
                        backoff_delay = min(backoff_delay * 2, BACKOFF_MAX)
                        continue
                    else:
                        backoff_delay = BACKOFF_BASE  # reset on success

                    tox_prob, cat_probs = toxicity_score(text, hf_bundle)
                    is_tox = tox_prob >= config.TOX_THRESHOLD
                    tox_counts[u]   += int(is_tox)
                    total_counts[u] += 1

                    post_rows.append({
                        "username": u,
                        "type":     "submission" if isinstance(
                                        act, praw.models.Submission) else "comment",
                        "id":            act.id,
                        "created_utc":   act.created_utc,
                        "title":         getattr(act, "title", None),
                        "body":          text,
                        "subreddit":     act.subreddit.display_name,
                        "score":         act.score,
                        "tox_score":     tox_prob,
                        "tox_cat0":      cat_probs[0],
                        "tox_cat1":      cat_probs[1],
                        "tox_cat2":      cat_probs[2],
                        "tox_cat3":      cat_probs[3],
                        "tox_cat4":      cat_probs[4],
                        "tox_cat5":      cat_probs[5],
                    })

                done_users.add(u)

                if len(post_rows) >= 10_000:
                    pd.DataFrame(post_rows).to_csv(
                        config.RAW_POSTS_CSV,
                        mode="a", index=False,
                        header=not os.path.exists(config.RAW_POSTS_CSV),
                    )
                    post_rows.clear()

    # ------------- flush remaining rows -------------
    if post_rows:
        pd.DataFrame(post_rows).to_csv(
            config.RAW_POSTS_CSV,
            mode="a", index=False,
            header=not os.path.exists(config.RAW_POSTS_CSV),
        )

    # ------------- write user labels -------------
    rows = []
    for u, total in total_counts.items():
        frac = tox_counts[u] / total
        rows.append({
            "username":  u,
            "posts":     total,
            "tox_posts": tox_counts[u],
            "tox_frac":  frac,
            "weak_label": int(frac >= config.USER_TOX_THRESHOLD),
        })
    pd.DataFrame(rows).to_csv(config.USER_LABELS_CSV, index=False)
    print(f"✓  collection finished – users: {len(rows)}")

# -------------------------------------------------
if __name__ == "__main__":
    main()
