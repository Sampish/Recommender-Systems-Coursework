#!/usr/bin/env python3
"""
run_eval.py - Comprehensive Evaluation Script for Hybrid Recommender Systems

Compares RS1 (TimeSVD++ + KNN) vs RS2 (SASRec + KNN)
Evaluates individual components and generates comparison visualizations.

Usage:
    python run_eval.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import load_npz
import torch

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

DATA_DIR = "KuaiRec 2-2.0/data"
MODEL_DIR = "models"
RESULTS_DIR = "evaluation_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Import models
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sasrec_model import SASRec

print("=" * 70)
print(" " * 10 + "COMPREHENSIVE EVALUATION SCRIPT")
print(" " * 5 + "RS1 (TimeSVD++ + KNN) vs RS2 (SASRec + KNN)")
print("=" * 70 + "\n")

# =========================
# LOAD DATA
# =========================
print("[1/6] Loading test data...")

def load_test_data():
    """Load test split for evaluation"""
    collab_df = pd.read_csv(os.path.join(DATA_DIR, "collab_df.csv"))
    content_df = pd.read_csv(os.path.join(DATA_DIR, "content_df.csv"))
    content_features = load_npz(os.path.join(DATA_DIR, "content_features.npz"))
    
    # Temporal split (80/20)
    if 'time' in collab_df.columns:
        collab_df = collab_df.sort_values(['user_id', 'time'])
    
    train_list, test_list = [], []
    for user_id, group in collab_df.groupby('user_id'):
        n = len(group)
        split_idx = int(n * 0.8)
        train_list.append(group.iloc[:split_idx])
        test_list.append(group.iloc[split_idx:])
    
    train_df = pd.concat(train_list)
    test_df = pd.concat(test_list)
    
    print(f"  ✅ Test: {len(test_df)} interactions, {test_df['user_id'].nunique()} users\n")
    return train_df, test_df, content_df, content_features

train_df, test_df, content_df, content_features = load_test_data()

# Calculate popularity (for serendipity)
item_counts = train_df['video_id'].value_counts()
total_users = train_df['user_id'].nunique()
item_popularity = (item_counts / total_users).to_dict()

def get_unexpectedness(vid):
    return 1.0 - item_popularity.get(vid, 0.0)

# =========================
# LOAD RS1 (TimeSVD++ + KNN)
# =========================
print("[2/6] Loading RS1 (TimeSVD++ + KNN)...")

rs1_path = os.path.join(MODEL_DIR, "hybrid_system_timesvdpp.pkl")
if not os.path.exists(rs1_path):
    print(f"  ⚠️ RS1 model not found at {rs1_path}")
    rs1_loaded = False
else:
    with open(rs1_path, 'rb') as f:
        rs1_system = pickle.load(f)
    print(f"  ✅ RS1 loaded\n")
    rs1_loaded = True

# =========================
# LOAD RS2 (SASRec + KNN)  
# =========================
print("[3/6] Loading RS2 (SASRec + KNN)...")

rs2_path = os.path.join(MODEL_DIR, "hybrid_sasrec_knn.pkl")
if not os.path.exists(rs2_path):
    print(f"  ⚠️ RS2 model not found at {rs2_path}")
    rs2_loaded = False
else:
    with open(rs2_path, 'rb') as f:
        rs2_system = pickle.load(f)
    
    # Load SASRec model state
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sasrec_model = SASRec(
        n_items=rs2_system['mappings']['n_items'],
        maxlen=rs2_system['mappings']['maxlen'],
        d_model=64,
        n_heads=2,
        n_blocks=2,
        dropout=0.2,
        use_time_features=True
    ).to(device)
    sasrec_model.load_state_dict(rs2_system['sasrec_model'])
    sasrec_model.eval()
    
    print(f"  ✅ RS2 loaded\n")
    rs2_loaded = True

if not rs1_loaded and not rs2_loaded:
    print("❌ No models found! Train models first.")
    sys.exit(1)

# =========================
# DEFINE SCORERS
# =========================
print("[4/6] Setting up evaluation scorers...")

# Shared KNN scorer
vid_to_content_idx = {vid: i for i, vid in enumerate(content_df['video_id'])}
content_idx_to_vid = {i: vid for vid, i in vid_to_content_idx.items()}

def knn_scorer(knn_model, user_id, items, train_history_map):
    """Score items using KNN"""
    history = train_history_map.get(user_id, [])
    if not history:
        return {vid: 0.0 for vid in items}
    
    hist_indices = [vid_to_content_idx[v] for v in history if v in vid_to_content_idx]
    if not hist_indices:
        return {vid: 0.0 for vid in items}
    
    user_profile = np.asarray(content_features[hist_indices].mean(axis=0))
    dists, indices = knn_model.kneighbors(user_profile, n_neighbors=200)
    sims = 1 - dists[0]
    
    scores = {vid: 0.0 for vid in items}
    for i, idx in enumerate(indices[0]):
        if idx in content_idx_to_vid:
            vid = content_idx_to_vid[idx]
            if vid in scores:
                scores[vid] = sims[i]
    return scores

def normalize_scores(scores_dict):
    """Normalize scores to [0, 1]"""
    vals = list(scores_dict.values())
    if not vals:
        return scores_dict
    mn, mx = min(vals), max(vals)
    if mx == mn:
        return {k: 0.0 for k in scores_dict}
    return {k: (v - mn) / (mx - mn) for k, v in scores_dict.items()}

print("  ✅ Scorers ready\n")

# =========================
# EVALUATE MODELS
# =========================
print("[5/6] Running evaluation...")

K_VALUES = [4, 10, 30, 50]
results = {}

test_users = list(test_df['user_id'].unique())[:100]  # Sample 100 users
test_grouped = test_df.groupby('user_id')['video_id'].apply(set).to_dict()
train_grouped = train_df.groupby('user_id')['video_id'].apply(set).to_dict()
all_items = content_df['video_id'].unique()

def evaluate_scorer(name, scorer_func):
    """Evaluate a scoring function"""
    metrics = {k: {'ndcg': [], 'hits': [], 'serendipity': []} for k in K_VALUES}
    
    for user_id in test_users:
        true_positives = test_grouped.get(user_id, set())
        if not true_positives:
            continue
        
        seen_items = train_grouped.get(user_id, set())
        scores_dict = scorer_func(user_id, all_items)
        
        candidates = [(vid, score) for vid, score in scores_dict.items() if vid not in seen_items]
        if not candidates:
            continue
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_recs = [x[0] for x in candidates]
        
        for k in K_VALUES:
            top_k = top_recs[:k]
            hits = [1 if item in true_positives else 0 for item in top_k]
            
            # NDCG
            dcg = np.sum([rel / np.log2(idx + 2) for idx, rel in enumerate(hits)])
            ideal_hits = sorted(hits, reverse=True)
            idcg = np.sum([rel / np.log2(idx + 2) for idx, rel in enumerate(ideal_hits)])
            metrics[k]['ndcg'].append((dcg / idcg) if idcg > 0 else 0.0)
            
            # Hit rate
            metrics[k]['hits'].append(sum(hits) / len(true_positives) if true_positives else 0.0)
            
            # Serendipity
            ser_val = sum(get_unexpectedness(item) for item in top_k if item in true_positives)
            metrics[k]['serendipity'].append(ser_val / k)
    
    # Average metrics
    for k in K_VALUES:
        for metric in ['ndcg', 'hits', 'serendipity']:
            metrics[k][metric] = np.mean(metrics[k][metric]) if metrics[k][metric] else 0.0
    
    return metrics

# Evaluate RS1 components
if rs1_loaded:
    print("  Evaluating RS1...")
    train_history_rs1 = rs1_system['train_history_map']
    knn_rs1 = rs1_system['knn_model']
    timesvd = rs1_system['timesvd_model']
    user_to_idx = rs1_system['user_to_idx']
    item_to_idx = rs1_system['item_to_idx']
    user_meta_lookup = rs1_system['user_meta_lookup']
    
    def get_alpha(user_id):
        if user_id not in user_meta_lookup.index:
            return 0.5
        row = user_meta_lookup.loc[user_id]
        if row['is_lowactive_period'] == 1:
            return 0.2
        deg = str(row['user_active_degree'])
        if deg == 'full_active':
            return 0.8
        if deg == 'high_active':
            return 0.7
        if deg == 'middle_active':
            return 0.6
        return 0.5
    
    # TimeSVD++ scorer
    def timesvd_scorer(user_id, items):
        scores = {}
        if user_id not in user_to_idx:
            return {vid: timesvd.global_mean for vid in items}
        
        user_idx = user_to_idx[user_id]
        user_history_items = train_history_rs1.get(user_id, [])
        user_history_indices = [item_to_idx[it] for it in user_history_items if it in item_to_idx]
        
        for vid in items:
            if vid in item_to_idx:
                item_idx = item_to_idx[vid]
                scores[vid] = timesvd.predict(user_idx, item_idx, user_history=user_history_indices)
            else:
                scores[vid] = timesvd.global_mean
        return scores
    
    # RS1 Hybrid
    def rs1_hybrid_scorer(user_id, items):
        c_scores = normalize_scores(knn_scorer(knn_rs1, user_id, items, train_history_rs1))
        l_scores = normalize_scores(timesvd_scorer(user_id, items))
        alpha = get_alpha(user_id)
        return {vid: alpha * l_scores.get(vid, 0) + (1 - alpha) * c_scores.get(vid, 0) for vid in items}
    
    results['RS1_TimeSVD++'] = evaluate_scorer("TimeSVD++", timesvd_scorer)
    results['RS1_KNN'] = evaluate_scorer("KNN", lambda u, i: knn_scorer(knn_rs1, u, i, train_history_rs1))
    results['RS1_Hybrid'] = evaluate_scorer("Hybrid RS1", rs1_hybrid_scorer)
    print("RS1 evaluated")

# Evaluate RS2 components  
if rs2_loaded:
    print("  Evaluating RS2...")
    
    # Load SASRec sequences for session mapping
    sasrec_dir = os.path.join(DATA_DIR, 'sasrec_preprocessed')
    with open(os.path.join(sasrec_dir, 'sequences.pkl'), 'rb') as f:
        seq_data = pickle.load(f)
    
    train_seqs_sasrec = seq_data['train']
    test_tgts_sasrec = seq_data['test_targets']
    mappings = rs2_system['mappings']
    
    # Create session_id to original user_id mapping
    # Session IDs are format: "userId_sessionIndex"
    session_to_user = {}
    for session_id in train_seqs_sasrec.keys():
        original_user_id = int(session_id.split('_')[0])
        if original_user_id not in session_to_user:
            session_to_user[original_user_id] = []
        session_to_user[original_user_id].append(session_id)
    
    # Use saved training history from RS2 model (if available)
    # This ensures consistency with how the model was actually trained
    if 'train_history_map' in rs2_system:
        train_history_rs2 = rs2_system['train_history_map']
        print("    → Using saved RS2 training history")
    else:
        print("    ⚠️  RS2 model missing train_history_map, using current train_grouped")
        print("       (This may cause inconsistency - retrain RS2 with fix!)")
        train_history_rs2 = train_grouped
    
    knn_rs2 = rs2_system['knn_model']
    
    # Get alpha function from user metadata
    user_meta_df = pd.read_csv(os.path.join(DATA_DIR, 'collab_df.csv'))
    user_meta_lookup2 = user_meta_df.groupby('user_id')[['user_active_degree', 'is_lowactive_period']].first()
    
    def get_alpha_rs2(user_id):
        if user_id not in user_meta_lookup2.index:
            return 0.5
        row = user_meta_lookup2.loc[user_id]
        if row['is_lowactive_period'] == 1:
            return 0.2
        deg = str(row['user_active_degree'])
        if deg == 'full_active':
            return 0.8
        if deg == 'high_active':
            return 0.7
        if deg == 'middle_active':
            return 0.6
        return 0.5
    
    # Proper SASRec scorer using trained model
    def sasrec_scorer(user_id, items):
        """Score items using SASRec model for a given user"""
        # Get user's sessions
        user_sessions = session_to_user.get(user_id, [])
        if not user_sessions:
            # User not in SASRec training, return neutral scores
            return {vid: 0.5 for vid in items}
        
        # Use most recent session (last one)
        session_id = user_sessions[-1]
        train_seq = train_seqs_sasrec.get(session_id, [])
        
        if not train_seq:
            return {vid: 0.5 for vid in items}
        
        # Pad sequence
        maxlen = mappings['maxlen']
        if len(train_seq) < maxlen:
            padded_seq = [0] * (maxlen - len(train_seq)) + train_seq
        else:
            padded_seq = train_seq[-maxlen:]
        
        # Convert to tensor
        seq_tensor = torch.LongTensor([padded_seq]).to(device)
        
        # Map items to SASRec item indices
        item2idx = mappings['item2idx']
        idx2item = mappings['idx2item']
        
        # Create candidate tensor (only items in SASRec vocab)
        candidate_indices = []
        candidate_vids = []
        for vid in items:
            if vid in item2idx:
                candidate_indices.append(item2idx[vid])
                candidate_vids.append(vid)
        
        if not candidate_indices:
            return {vid: 0.5 for vid in items}
        
        candidates_tensor = torch.LongTensor([candidate_indices]).to(device)
        
        # Predict scores
        with torch.no_grad():
            scores = sasrec_model.predict(seq_tensor, candidates_tensor).squeeze(0).cpu().numpy()
        
        # Create score dictionary
        score_dict = {}
        for vid, score in zip(candidate_vids, scores):
            score_dict[vid] = float(score)
        
        # For items not in SASRec vocab, use neutral score
        for vid in items:
            if vid not in score_dict:
                score_dict[vid] = 0.5
        
        return score_dict
    
    # RS2 KNN (same approach as RS1)
    def rs2_knn_scorer(user_id, items):
        return knn_scorer(knn_rs2, user_id, items, train_history_rs2)
    
    # RS2 Hybrid (SASRec + KNN)
    def rs2_hybrid_scorer(user_id, items):
        s_scores = normalize_scores(sasrec_scorer(user_id, items))
        c_scores = normalize_scores(rs2_knn_scorer(user_id, items))
        alpha = get_alpha_rs2(user_id)
        return {vid: alpha * s_scores.get(vid, 0) + (1 - alpha) * c_scores.get(vid, 0) for vid in items}
    
    results['RS2_SASRec'] = evaluate_scorer("SASRec", sasrec_scorer)
    results['RS2_KNN'] = evaluate_scorer("KNN", rs2_knn_scorer)
    results['RS2_Hybrid'] = evaluate_scorer("Hybrid RS2", rs2_hybrid_scorer)
    print("    ✅ RS2 evaluated with SASRec model")

print("  ✅ Evaluation complete\n")



# =========================
# GENERATE VISUALIZATIONS
# =========================
print("[6/6] Generating visualizations...")

# Create comparison table
comparison_data = []
for model_name, metrics in results.items():
    for k in K_VALUES:
        comparison_data.append({
            'Model': model_name,
            'K': k,
            'NDCG': metrics[k]['ndcg'],
            'Hit Rate': metrics[k]['hits'],
            'Serendipity': metrics[k]['serendipity']
        })

df_results = pd.DataFrame(comparison_data)

# Print table
print("\n" + "=" * 70)
print(" " * 20 + "EVALUATION RESULTS")
print("=" * 70 + "\n")
print(df_results.to_string(index=False))
print("\n" + "=" * 70 + "\n")

# Plot comparisons
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

metrics_to_plot = ['NDCG', 'Hit Rate', 'Serendipity']
for idx, metric in enumerate(metrics_to_plot):
    ax = axes[idx]
    
    for model_name in results.keys():
        model_data = df_results[df_results['Model'] == model_name]
        ax.plot(model_data['K'], model_data[metric], marker='o', label=model_name, linewidth=2)
    
    ax.set_xlabel('K (Top-K Recommendations)', fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(RESULTS_DIR, 'model_comparison.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"  ✅ Saved comparison plot: {plot_path}")

# Save results to CSV
csv_path = os.path.join(RESULTS_DIR, 'evaluation_results.csv')
df_results.to_csv(csv_path, index=False)
print(f"  ✅ Saved results table: {csv_path}\n")

print("=" * 70)
print(" " * 20 + "EVALUATION COMPLETE!")
print("=" * 70)
print(f"\nResults saved to {RESULTS_DIR}/")
print("  - evaluation_results.csv")
print("  - model_comparison.png\n")
