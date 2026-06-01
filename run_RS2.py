"""
Interactive CLI Recommender System - RS2 (SASRec + KNN)
Run with: python run_RS2.py
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
import getpass
from term_image.image import from_file
import torch

# ANSI color codes for better UI
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ORANGE = '\033[38;5;208m'  # Orange color
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    WHITE = '\033[97m'

def clear_screen():
    """Clear the terminal screen"""
    os.system('clear' if os.name == 'posix' else 'cls')

def print_banner():
    """Display welcome banner"""
    clear_screen()
    print(Colors.ORANGE + "=" * 70)
    print("||" + " " * 66 + "||")
    print("||" + " " * 15 + Colors.BOLD + "KuaiShou Recommender System" + Colors.END + Colors.ORANGE + " " * 24 + "||")
    print("||" + " " * 66 + "||")
    print("=" * 70 + Colors.END)
    print()

def login():
    """Handle user login"""
    print_banner()
    print(Colors.BOLD + "Please login to continue:\n" + Colors.END)
    
    # Load user IDs to validate
    collab_df = pd.read_csv('KuaiRec 2-2.0/data/collab_df.csv')
    valid_users = set(collab_df['user_id'].unique())
    
    while True:
        username = input(Colors.BLUE + "Username (User ID): " + Colors.END)
        
        try:
            user_id = int(username)
            if user_id in valid_users:
                break
            else:
                print(Colors.RED + f"User ID {user_id} not found. Please try again.\n" + Colors.END)
        except ValueError:
            print(Colors.RED + "Please enter a valid numeric user ID.\n" + Colors.END)
    
    password = getpass.getpass(Colors.BLUE + "Password: " + Colors.END)
    
    # Simulate authentication (password is cosmetic)
    print(Colors.GREEN + "\nLogin successful! Welcome, User " + str(user_id) + Colors.END)
    
    return user_id

def load_system():
    """Load the hybrid recommender system and data"""
    print(Colors.YELLOW + "\nLoading recommender system..." + Colors.END)
    
    # Load hybrid model (SASRec + KNN)
    with open('models/hybrid_sasrec_knn.pkl', 'rb') as f:
        system = pickle.load(f)
    
    # Load enhanced content data with synthetic titles and thumbnails
    content_df = pd.read_csv('KuaiRec 2-2.0/data/content_df_cli.csv')
    
    # Load collaborative data
    collab_df = pd.read_csv('KuaiRec 2-2.0/data/collab_df.csv')
    
    print(Colors.GREEN + "System loaded successfully!" + Colors.END)
    
    return system, content_df, collab_df

def get_dynamic_alpha(user_id, user_meta):
    """Calculate dynamic alpha for hybrid weighting"""
    if user_id not in user_meta.index:
        return 0.5
    
    row = user_meta.loc[user_id]
    
    # Low activity period check
    if row.get('is_lowactive_period', 0) == 1:
        return 0.2
    
    # Activity degree check
    degree = str(row.get('user_active_degree', 'middle_active'))
    
    if degree == 'full_active':
        return 0.8
    elif degree == 'high_active':
        return 0.7
    elif degree == 'middle_active':
        return 0.6
    else:
        return 0.5

def get_recommendations(system, user_id, content_df, collab_df, top_k=100):
    """Get recommendations for a user using the hybrid model (SASRec + KNN)"""
    
    knn_model = system['knn_model']
    sasrec_model = system['sasrec_model']
    user_meta = system.get('user_meta', pd.DataFrame())
    content_features = system['content_features']
    user_to_idx = system.get('user_to_idx', {})
    item_to_idx = system.get('item_to_idx', {})
    idx_to_item = system.get('idx_to_item', {})
    user_sequences = system.get('user_sequences', {})
    
    # Get user's watch history
    history = collab_df[collab_df['user_id'] == user_id]['video_id'].tolist()
    
    if len(history) == 0:
        print(Colors.RED + "No watch history found for this user." + Colors.END)
        return []
    
    # Get alpha for hybrid weighting
    alpha = get_dynamic_alpha(user_id, user_meta)
    
    # 1. Content-based scores (KNN)
    content_scores = {}
    watched_indices = content_df[content_df['video_id'].isin(history)].index.tolist()
    
    if watched_indices and len(watched_indices) > 0:
        user_profile = np.asarray(content_features[watched_indices].mean(axis=0))
        dists, indices = knn_model.kneighbors(user_profile, n_neighbors=min(200, content_features.shape[0]))
        sims = 1 - dists[0]
        
        for i, idx in enumerate(indices[0]):
            vid = content_df.iloc[idx]['video_id']
            content_scores[vid] = sims[i]
    
    # 2. Collaborative scores (SASRec)
    collab_scores = {}
    if user_id in user_sequences:
        sequence = user_sequences[user_id]
        
        # Prepare sequence for SASRec
        max_len = sasrec_model.max_len
        if len(sequence) > max_len:
            sequence = sequence[-max_len:]
        
        # Pad sequence
        padded_seq = [0] * (max_len - len(sequence)) + sequence
        seq_tensor = torch.LongTensor([padded_seq])
        
        # Get SASRec predictions
        sasrec_model.eval()
        with torch.no_grad():
            predictions = sasrec_model(seq_tensor)  # Shape: [1, max_len, num_items]
            predictions = predictions[0, -1, :].numpy()  # Get last position predictions
        
        # Map predictions to video IDs
        for item_idx, score in enumerate(predictions):
            if item_idx in idx_to_item:
                vid = idx_to_item[item_idx]
                collab_scores[vid] = score
        
        # Normalize SASRec scores
        if len(collab_scores) > 0:
            scores_array = np.array(list(collab_scores.values()))
            if scores_array.max() > scores_array.min():
                normalized_scores = (scores_array - scores_array.min()) / (scores_array.max() - scores_array.min())
                collab_scores = {vid: normalized_scores[i] for i, vid in enumerate(collab_scores.keys())}
    
    # 3. Combine scores
    final_scores = {}
    candidates = set(content_scores.keys()) | set(collab_scores.keys())
    candidates -= set(history)  # Remove already watched
    
    for vid in candidates:
        c = content_scores.get(vid, 0)
        l = collab_scores.get(vid, 0)
        final_scores[vid] = alpha * l + (1 - alpha) * c
    
    # Get top K
    top_items = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    # Get video metadata
    recommendations = []
    for vid, score in top_items:
        video_data = content_df[content_df['video_id'] == vid]
        if not video_data.empty:
            video_info = video_data.iloc[0]
            
            # Get tags (video_tag_name column)
            tags = video_info.get('video_tag_name', 'N/A')
            if pd.notna(tags) and isinstance(tags, str):
                # Clean up tags if needed
                tags = tags.strip()
            else:
                tags = 'N/A'
            
            recommendations.append({
                'video_id': vid,
                'title': video_info.get('synthetic_title', 'Unknown'),
                'category': video_info.get('first_level_category_name', 'N/A'),
                'thumbnail': video_info.get('thumbnail_path', ''),
                'tags': tags,
                'author_id': video_info.get('author_id', 'N/A'),
                'like_cnt': int(video_info.get('like_cnt', 0)) if pd.notna(video_info.get('like_cnt')) else 0,
                'score': score
            })
    
    return recommendations

def display_recommendations(recommendations, page=0, page_size=4):
    """Display a page of recommendations with horizontal separators only"""
    clear_screen()
    
    start_idx = page * page_size
    end_idx = min(start_idx + page_size, len(recommendations))
    current_recs = recommendations[start_idx:end_idx]
    
    if not current_recs:
        print(Colors.RED + "No recommendations to display." + Colors.END)
        return
    
    WIDTH = 80
    
    # Top header
    print(Colors.ORANGE + "=" * WIDTH + Colors.END)
    print(Colors.ORANGE + Colors.BOLD + "KuaiShou".center(WIDTH) + Colors.END)
    print(Colors.ORANGE + "=" * WIDTH + Colors.END)
    print()
    print(Colors.YELLOW + f"Showing {start_idx + 1}-{end_idx} of {len(recommendations)} recommendations\n" + Colors.END)
    
    for i, rec in enumerate(current_recs, 1):
        # Recommendation separator
        print(Colors.BOLD + "=" * WIDTH + Colors.END)
        print(Colors.WHITE + f"Recommendation #{start_idx + i}" + Colors.END)
        print("=" * WIDTH)
        
        # Display thumbnail image
        thumbnail_path = rec['thumbnail']
        if thumbnail_path and os.path.exists(thumbnail_path):
            try:
                img = from_file(thumbnail_path)
                img.height = 15
                print(img)
            except Exception as e:
                print(Colors.YELLOW + "[Image not available]" + Colors.END)
        else:
            print(Colors.YELLOW + "[No thumbnail]" + Colors.END)
        
        # Display metadata
        print()
        print(Colors.BOLD + f"Title: {rec['title']}" + Colors.END)
        print(f"Tags: {rec['tags']}")
        print(f"Author ID: {rec['author_id']}")
        print(f"Likes: {rec['like_cnt']}")
        print()
    
    # Bottom separator
    print(Colors.BOLD + Colors.WHITE + "=" * WIDTH + Colors.END)
    print()
def navigation_menu(total_recommendations, page, page_size=4):
    """Display navigation options and get user input"""
    total_pages = (total_recommendations + page_size - 1) // page_size
    
    print(Colors.GREEN + "Navigation:" + Colors.END)
    
    if page < total_pages - 1:
        print("  [N] Next page")
    if page > 0:
        print("  [P] Previous page")
    print("  [Q] Quit")
    
    print()
    choice = input(Colors.BLUE + "Your choice: " + Colors.END).strip().lower()
    
    return choice

def main():
    """Main CLI application loop"""
    try:
        # Login
        user_id = login()
        
        # Load system
        system, content_df, collab_df = load_system()
        
        # Get recommendations
        print(Colors.YELLOW + f"\nGenerating personalized recommendations for User {user_id}...\n" + Colors.END)
        recommendations = get_recommendations(system, user_id, content_df, collab_df, top_k=100)
        
        if not recommendations:
            print(Colors.RED + "\nNo recommendations available for this user." + Colors.END)
            return
        
        print(Colors.GREEN + f"Found {len(recommendations)} recommendations!\n" + Colors.END)
        
        # Auto-proceed to display (no "Press Enter" prompt)
        import time
        time.sleep(1)  # Brief pause for user to read the message
        
        # Display loop with pagination
        page = 0
        page_size = 4
        
        while True:
            display_recommendations(recommendations, page, page_size)
            choice = navigation_menu(len(recommendations), page, page_size)
            
            if choice == 'n' and page < (len(recommendations) + page_size - 1) // page_size - 1:
                page += 1
            elif choice == 'p' and page > 0:
                page -= 1
            elif choice == 'q':
                clear_screen()
                print(Colors.CYAN + "\nThank you for using KuaiShou Recommender System!\n" + Colors.END)
                break
            else:
                print(Colors.RED + "\nInvalid choice or no more pages in that direction.\n" + Colors.END)
                input("Press Enter to continue...")
    
    except KeyboardInterrupt:
        clear_screen()
        print(Colors.YELLOW + "\n\nProgram interrupted by user.\n" + Colors.END)
        sys.exit(0)
    except Exception as e:
        print(Colors.RED + f"\nError: {e}" + Colors.END)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
