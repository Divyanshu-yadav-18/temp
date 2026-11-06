import re
import math

text1 = "india lost the cricket match yesterday between india and south africa"
text2 = "Pakistan won the hockey match between India and Pakistan yesterday."

# NORMALIZATION

def normalize_text(text):
    """
    Normalizes input text by:
    1. Converting to lowercase
    2. Removing punctuation
    3. Tokenizing into words
    """
    
    lowercased = text.lower()

    cleaned = re.sub(r'[^\w\s]', '', lowercased)

    tokens = cleaned.split()

    return tokens

def cosine_similarity(text1, text2):
    """
    Calculates cosine similarity between two text strings using Term Frequency (TF).
    Formula: (A · B) / (||A|| * ||B||)
    """

    tokens1 = normalize_text(text1)
    tokens2 = normalize_text(text2)

    vocab = list(set(tokens1 + tokens2))
    vocab.sort()

    vec1 = [tokens1.count(word) for word in vocab]
    vec2 = [tokens2.count(word) for word in vocab]
    dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))

    magnitude1 = math.sqrt(sum(v1 ** 2 for v1 in vec1))
    magnitude2 = math.sqrt(sum(v2 ** 2 for v2 in vec2))

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0, vocab, vec1, vec2
        
    similarity = dot_product / (magnitude1 * magnitude2)
    return similarity, vocab, vec1, vec2

if __name__ == '__main__':
    print("==========================================")
    print("Q2: NORMALIZATION RESULTS")
    print("==========================================")
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)
    
    print(f"Text 1 Tokens:\n{norm1}\n")
    print(f"Text 2 Tokens:\n{norm2}\n")
    
    print("==========================================")
    print("Q1: COSINE SIMILARITY RESULTS")
    print("==========================================")
    similarity, vocab, vec1, vec2 = cosine_similarity(text1, text2)
    
    print("Vocabulary List:")
    print(vocab)
    print("\nVector 1 (Text 1):", vec1)
    print("Vector 2 (Text 2):", vec2)
    print(f"\nCosine Similarity Score: {similarity:.4f}")
