#!/usr/bin/env python
# 02_data_analysis.py
# Purpose: Perform exploratory data analysis on the X-ray dataset

import os
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
import config

def analyze_caption_lengths(captions):
    """Analyze caption lengths in characters and words"""
    caption_lengths = [len(caption) for caption in captions]
    caption_word_counts = [len(caption.split()) for caption in captions]
    
    length_stats = {
        "char_min": min(caption_lengths),
        "char_max": max(caption_lengths),
        "char_mean": sum(caption_lengths) / len(caption_lengths),
        "char_median": np.median(caption_lengths),
        "word_min": min(caption_word_counts),
        "word_max": max(caption_word_counts),
        "word_mean": sum(caption_word_counts) / len(caption_word_counts),
        "word_median": np.median(caption_word_counts)
    }
    
    print(f"Caption length statistics:")
    print(f"  Characters: min={length_stats['char_min']}, max={length_stats['char_max']}, "
          f"mean={length_stats['char_mean']:.2f}, median={length_stats['char_median']:.2f}")
    print(f"  Words: min={length_stats['word_min']}, max={length_stats['word_max']}, "
          f"mean={length_stats['word_mean']:.2f}, median={length_stats['word_median']:.2f}")
    
    # Plot length distributions
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(caption_lengths, bins=30, alpha=0.7, color='blue')
    plt.title('Distribution of Caption Character Lengths')
    plt.xlabel('Character Count')
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(caption_word_counts, bins=30, alpha=0.7, color='green')
    plt.title('Distribution of Caption Word Counts')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_DIR, 'caption_length_distributions.png'))
    
    return length_stats

def analyze_vocabulary(captions):
    """Analyze vocabulary and common terms in captions"""
    # Combine all captions
    all_text = " ".join(captions)
    
    # Extract words and clean them
    words = re.findall(r'\b\w+\b', all_text.lower())
    
    # Define stopwords to filter out
    stopwords = set(STOPWORDS)
    medical_stopwords = {'the', 'and', 'of', 'is', 'in', 'with', 'are', 'to', 
                         'a', 'no', 'or', 'for', 'on', 'at', 'be', 'this', 
                         'that', 'by', 'an'}
    stopwords.update(medical_stopwords)
    
    # Count words excluding stopwords and short words
    filtered_words = [word for word in words if word not in stopwords and len(word) > 2]
    word_counts = Counter(filtered_words)
    
    # Get most common words
    most_common = word_counts.most_common(30)
    
    # Plot most common words
    plt.figure(figsize=(12, 8))
    words, counts = zip(*most_common[:20])  # Top 20 for readability
    plt.barh(words, counts, color='skyblue')
    plt.title('Most Common Medical Terms in Captions')
    plt.xlabel('Occurrences')
    plt.ylabel('Term')
    plt.gca().invert_yaxis()  # Display highest counts at top
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_DIR, 'common_terms.png'))
    
    # Create word cloud
    wc = WordCloud(width=800, height=400, background_color='white',
                  stopwords=stopwords, max_words=100, colormap='viridis')
    wc.generate(all_text)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_DIR, 'caption_wordcloud.png'))
    
    # Save most common words to CSV
    common_words_df = pd.DataFrame(most_common, columns=['Term', 'Occurrences'])
    common_words_df.to_csv(os.path.join(config.RESULTS_DIR, 'common_terms.csv'), index=False)
    
    print(f"Top 10 most common medical terms:")
    for word, count in most_common[:10]:
        print(f"  {word}: {count} occurrences")
        
    return most_common

def main():
    """Perform exploratory data analysis on the dataset"""
    # Create necessary directories
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    # Load dataset splits
    dataset_path = os.path.join(config.DATA_DIR, "dataset_splits.json")
    
    try:
        with open(dataset_path, 'r') as f:
            splits = json.load(f)
            
        train_samples = splits['train']
        val_samples = splits['validation']
        test_samples = splits['test']
    except (FileNotFoundError, KeyError):
        print(f"Dataset splits not found at {dataset_path}")
        print("Please run 01_data_load.py first")
        return
    
    # Combine all captions for analysis
    all_captions = []
    for sample in train_samples + val_samples + test_samples:
        all_captions.append(sample['caption'])
    
    print(f"\n--- Exploratory Data Analysis ---")
    print(f"Analyzing {len(all_captions)} captions...")
    
    # Analyze caption lengths
    length_stats = analyze_caption_lengths(all_captions)
    
    # Analyze vocabulary
    common_terms = analyze_vocabulary(all_captions)
    
    # Save analysis results
    analysis_results = {
        "length_stats": length_stats,
        "common_terms": common_terms[:50]  # Save top 50 terms
    }
    
    with open(os.path.join(config.RESULTS_DIR, 'eda_results.json'), 'w') as f:
        # Convert terms to list of lists since tuples aren't JSON serializable
        analysis_results["common_terms"] = [[term, count] for term, count in analysis_results["common_terms"]]
        json.dump(analysis_results, f, indent=2)
    
    print(f"\nAnalysis complete. Results saved to {config.RESULTS_DIR}")
    print(f"Visualizations saved to {config.FIGURES_DIR}")

if __name__ == "__main__":
    main()
