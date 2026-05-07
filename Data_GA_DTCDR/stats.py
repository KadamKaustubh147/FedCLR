import pandas as pd
import numpy as np

def analyze_dat_file(filepath):
    """
    Try different formats to read .dat files
    """
    print(f"\nAnalyzing: {filepath}")
    print("-" * 40)
    
    # Try different separators
    for sep in ['\t', ',', ' ', '::', '::']:
        try:
            df = pd.read_csv(filepath, sep=sep, header=None, engine='python')
            print(f"  Read with sep='{sep}': {df.shape[0]} rows, {df.shape[1]} cols")
            print(f"  First 3 rows:\n{df.head(3)}")
            print(f"  Column dtypes: {df.dtypes.tolist()}")
            break
        except Exception as e:
            continue
    
    return df

def get_stats(df, user_col=0, item_col=1, rating_col=2):
    """
    Print dataset statistics
    """
    print(f"\n  === Stats ===")
    print(f"  Users:        {df[user_col].nunique()}")
    print(f"  Items:        {df[item_col].nunique()}")
    print(f"  Interactions: {len(df)}")
    print(f"  Rating range: {df[rating_col].min()} - {df[rating_col].max()}")
    
    density = len(df) / (df[user_col].nunique() * df[item_col].nunique())
    print(f"  Density:      {density:.6f}")
    print(f"  Sparsity:     {1-density:.6f}")
    
    print(f"\n  Avg interactions per user: {len(df)/df[user_col].nunique():.1f}")
    print(f"  Avg interactions per item: {len(df)/df[item_col].nunique():.1f}")
    
    print(f"\n  Rating distribution:")
    print(df[rating_col].value_counts().sort_index())

def main():
    import os
    
    # ============================================================
    # CHANGE THESE PATHS TO YOUR DOWNLOADED .dat FILES
    # ============================================================
    files = {
        "movie": "douban_movie/ratings.dat",   # change to actual filename
        "music": "douban_music/ratings.dat",   # change to actual filename
        "book":  "douban_book/ratings.dat",    # change to actual filename
    }

    dfs = {}
    for domain, filepath in files.items():
        if not os.path.exists(filepath):
            print(f"File not found: {filepath} — skipping")
            continue
        
        df = analyze_dat_file(filepath)
        get_stats(df)
        dfs[domain] = df

    # =========================
    # OVERLAPPING USERS
    # =========================
    if "movie" in dfs and "music" in dfs:
        movie_users = set(dfs["movie"][0].unique())
        music_users = set(dfs["music"][0].unique())
        overlap_movie_music = movie_users & music_users
        print(f"\n=== Overlapping Users ===")
        print(f"Movie & Music: {len(overlap_movie_music)}  (paper target: 1666)")

    if "movie" in dfs and "book" in dfs:
        movie_users = set(dfs["movie"][0].unique())
        book_users  = set(dfs["book"][0].unique())
        overlap_movie_book = movie_users & book_users
        print(f"Movie & Book:  {len(overlap_movie_book)}  (paper target: 2106)")

    if "music" in dfs and "book" in dfs:
        music_users = set(dfs["music"][0].unique())
        book_users  = set(dfs["book"][0].unique())
        overlap_music_book = music_users & book_users
        print(f"Music & Book:  {len(overlap_music_book)}  (paper target: 1566)")

if __name__ == "__main__":
    main()