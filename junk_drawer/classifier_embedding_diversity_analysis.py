#!/usr/bin/env python3
"""
Analyze diversity of calendar tool examples using text embeddings.
"""

import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import statistics

def load_examples(file_path):
    """Load examples from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def analyze_diversity_tfidf(queries):
    """Analyze diversity using TF-IDF vectors."""
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    vectors = vectorizer.fit_transform(queries)
    
    # Compute pairwise cosine similarities
    similarities = cosine_similarity(vectors)
    
    # Get upper triangle (excluding diagonal)
    n = len(queries)
    similarities_list = []
    for i in range(n):
        for j in range(i+1, n):
            similarities_list.append(similarities[i][j])
    
    return similarities_list

def analyze_diversity_sentence_transformer(queries):
    """Analyze diversity using sentence transformers."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(queries)
    
    # Compute pairwise cosine similarities
    similarities = cosine_similarity(embeddings)
    
    # Get upper triangle (excluding diagonal)  
    n = len(queries)
    similarities_list = []
    for i in range(n):
        for j in range(i+1, n):
            similarities_list.append(similarities[i][j])
    
    return similarities_list

def print_diversity_stats(similarities, method_name):
    """Print diversity statistics."""
    print(f"\n{method_name} Diversity Analysis:")
    print(f"Mean similarity: {statistics.mean(similarities):.3f}")
    print(f"Median similarity: {statistics.median(similarities):.3f}")
    print(f"Std dev similarity: {statistics.stdev(similarities):.3f}")
    print(f"Min similarity: {min(similarities):.3f}")
    print(f"Max similarity: {max(similarities):.3f}")
    
    # Diversity score (1 - mean similarity)
    diversity_score = 1 - statistics.mean(similarities)
    print(f"Diversity score: {diversity_score:.3f}")
    
    # Count high similarity pairs (>0.7)
    high_sim_count = sum(1 for s in similarities if s > 0.7)
    print(f"High similarity pairs (>0.7): {high_sim_count}")

def analyze_file(file_path):
    """Analyze diversity for a single file."""
    try:
        examples = load_examples(file_path)
        # Handle both list format and lines with comments
        if isinstance(examples, list) and len(examples) > 0:
            # Skip comment lines that start with //
            if isinstance(examples[0], str):
                queries = [line.strip() for line in examples if not line.strip().startswith('//') and line.strip()]
            else:
                queries = [example['query'] for example in examples if 'query' in example]
        else:
            return None
            
        if len(queries) < 2:
            return None
            
        # Sentence transformer analysis
        st_similarities = analyze_diversity_sentence_transformer(queries)
        
        return {
            'file': file_path.split('/')[-1],
            'count': len(queries),
            'mean_similarity': statistics.mean(st_similarities),
            'diversity_score': 1 - statistics.mean(st_similarities),
            'min_similarity': min(st_similarities),
            'max_similarity': max(st_similarities),
            'std_similarity': statistics.stdev(st_similarities),
            'high_sim_pairs': sum(1 for s in st_similarities if s > 0.7),
            'queries': queries
        }
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
        return None

def main():
    import os
    
    calendar_dir = '/Users/taylut/Programming/GitHub/botwithmemory/classoutput_temp/calendar'
    json_files = [f for f in os.listdir(calendar_dir) if f.endswith('.json')]
    
    print(f"Analyzing {len(json_files)} calendar tool files...")
    
    results = []
    all_queries = []
    
    for json_file in sorted(json_files):
        file_path = os.path.join(calendar_dir, json_file)
        result = analyze_file(file_path)
        if result:
            results.append(result)
            all_queries.extend(result['queries'])
            
            print(f"\n{result['file']}:")
            print(f"  Examples: {result['count']}")
            print(f"  Diversity score: {result['diversity_score']:.3f}")
            print(f"  Mean similarity: {result['mean_similarity']:.3f}")
            print(f"  Range: {result['min_similarity']:.3f} - {result['max_similarity']:.3f}")
            print(f"  High similarity pairs (>0.7): {result['high_sim_pairs']}")
    
    # Overall analysis across all files
    if len(all_queries) > 1:
        print(f"\n{'='*50}")
        print("OVERALL CALENDAR TOOL ASSESSMENT")
        print(f"{'='*50}")
        
        overall_similarities = analyze_diversity_sentence_transformer(all_queries)
        overall_diversity = 1 - statistics.mean(overall_similarities)
        
        print(f"Total examples across all files: {len(all_queries)}")
        print(f"Overall diversity score: {overall_diversity:.3f}")
        print(f"Overall mean similarity: {statistics.mean(overall_similarities):.3f}")
        print(f"Overall range: {min(overall_similarities):.3f} - {max(overall_similarities):.3f}")
        print(f"Overall high similarity pairs (>0.7): {sum(1 for s in overall_similarities if s > 0.7)}")
        
        # Summary assessment
        print(f"\nASSESSMENT:")
        if overall_diversity > 0.7:
            print("✅ EXCELLENT diversity - great for embeddings classification")
        elif overall_diversity > 0.6:
            print("✅ GOOD diversity - suitable for embeddings classification")
        elif overall_diversity > 0.5:
            print("⚠️  MODERATE diversity - may need more varied examples")
        else:
            print("❌ LOW diversity - needs more varied examples")
            
        # File-level summary
        print(f"\nFILE BREAKDOWN:")
        for result in sorted(results, key=lambda x: x['diversity_score'], reverse=True):
            status = "✅" if result['diversity_score'] > 0.6 else "⚠️" if result['diversity_score'] > 0.5 else "❌"
            print(f"{status} {result['file']}: {result['diversity_score']:.3f} ({result['count']} examples)")

if __name__ == "__main__":
    main()