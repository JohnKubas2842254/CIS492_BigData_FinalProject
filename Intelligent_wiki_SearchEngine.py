import json
import nltk #type: ignore
import re
from nltk.corpus import stopwords #type: ignore
from nltk.stem import WordNetLemmatizer #type: ignore
from collections import defaultdict, Counter
import pymysql #type: ignore
import time
import math
from nltk.util import ngrams # type: ignore

# --- Database Configuration ---
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "2842254K", 
    "database": "BigData_FinalProject",
    "charset": "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor # Use DictCursor for easy row access
}

# --- NLP Pipeline (Copied EXACTLY from JSON_to_InvertedIndex.py) ---
# Initialize lemmatizer and stopwords list
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Ensure NLTK data is available (run downloads if necessary, same as indexer)
# try: nltk.data.find('corpora/wordnet')
# except: nltk.download('wordnet')
# try: nltk.data.find('corpora/stopwords')
# except: nltk.download('stopwords')
# try: nltk.data.find('tokenizers/punkt')
# except: nltk.download('punkt')
# try: nltk.data.find('taggers/averaged_perceptron_tagger')
# except: nltk.download('averaged_perceptron_tagger')


def nlp_pipeline(text):
    """
    Applies NLP preprocessing steps including n-gram extraction:
    (Exact copy from JSON_to_InvertedIndex.py)
    """
    if not isinstance(text, str):
        return {'tokens': [], 'pos_tags': []}

    text_lower = text.lower()
    text_cleaned = re.sub(r'[^a-z0-9\s]', '', text_lower)
    tokens = nltk.word_tokenize(text_cleaned)

    if not tokens:
        return {'tokens': [], 'pos_tags': []}

    pos_tags = nltk.pos_tag(tokens)
    final_tokens_for_index = []

    # Unigram processing
    for word, tag in pos_tags:
        if word in stop_words:
            continue
        lemma = lemmatizer.lemmatize(word)
        if lemma.isalpha() and len(lemma) > 1:
            final_tokens_for_index.append(lemma)

    # N-gram generation and filtering
    allowed_bigram_patterns = [
        ('JJ', 'NN'), ('JJ', 'NNS'), ('NN', 'NN'), ('NNS', 'NN'), ('NN', 'NNS'),
        ('NNS', 'NNS'), ('NNP', 'NNP'), ('NNPS', 'NNP'), ('NNP', 'NNPS'),
        ('NNPS', 'NNPS'), ('NNP', 'CD'), ('NN', 'CD')
    ]
    allowed_trigram_patterns = [
        ('JJ', 'NN', 'NN'), ('JJ', 'NNS', 'NN'), ('JJ', 'NN', 'NNS'),
        ('NN', 'NN', 'NN'), ('NNS', 'NN', 'NN'), ('NN', 'NNS', 'NN'),
        ('NN', 'NN', 'NNS'), ('NNP', 'NNP', 'NNP'), ('NNP', 'NNP', 'CD'),
        ('NNP', 'NN', 'NN'), ('NN', 'IN', 'NN')
    ]

    # Process Bi-grams
    for gram in ngrams(pos_tags, 2):
        tags = tuple(tag for word, tag in gram)
        tag_prefixes = tuple(t[:2] for t in tags)
        if tag_prefixes in allowed_bigram_patterns:
            words = [word for word, tag in gram]
            lemmatized_words = [lemmatizer.lemmatize(w) for w in words]
            if all(lemmatized_words):
                phrase = "_".join(lemmatized_words)
                final_tokens_for_index.append(phrase)

    # Process Tri-grams
    for gram in ngrams(pos_tags, 3):
        tags = tuple(tag for word, tag in gram)
        tag_prefixes = tuple(t[:2] for t in tags)
        if tag_prefixes in allowed_trigram_patterns:
            words = [word for word, tag in gram]
            lemmatized_words = [lemmatizer.lemmatize(w) for w in words]
            if all(lemmatized_words):
                phrase = "_".join(lemmatized_words)
                final_tokens_for_index.append(phrase)

    return {
        'tokens': final_tokens_for_index,
        'pos_tags': pos_tags # Keep POS tags if needed later, not used in current search logic
    }
# --- End of NLP Pipeline ---

def search_documents(query_text, top_n=10):
    """
    Processes a query, calculates TF-IDF scores, and returns top N documents.
    """
    start_time = time.time()

    # 1. Process the query using the same NLP pipeline
    nlp_result = nlp_pipeline(query_text)
    query_terms = nlp_result['tokens'] # Use the processed tokens (unigrams + ngrams)

    if not query_terms:
        print("Query could not be processed or resulted in no valid terms.")
        return []

    print(f"Processed query terms: {query_terms}")

    doc_scores = defaultdict(float)
    total_docs_in_collection = 0

    try:
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # 2. Get total number of documents (N) from IndexMetadata table
        cursor.execute("SELECT Value FROM IndexMetadata WHERE KeyName = 'TotalDocuments'")
        result = cursor.fetchone()
        if result and result['Value'] > 0:
            total_docs_in_collection = result['Value']
            print(f"Retrieved TotalDocuments (N) = {total_docs_in_collection}")
        else:
            print("Error: Could not retrieve TotalDocuments count from IndexMetadata. Aborting search.")
            cursor.close()
            conn.close()
            return [] # Cannot proceed without N

        if total_docs_in_collection <= 1:
             print("Warning: Total document count is very low. IDF scores may not be meaningful.")

        # 3. Calculate scores for documents containing query terms
        for term in set(query_terms): # Use set to process each unique term once
            # Fetch term frequency in the collection (df) and IDF
            cursor.execute("SELECT Term, TotalDocsFreq FROM Dictionary WHERE Term = %s", (term[:255],)) # Ensure term is truncated
            dict_entry = cursor.fetchone()

            if dict_entry:
                doc_freq = dict_entry['TotalDocsFreq']
                if doc_freq > 0:
                    idf = math.log10(total_docs_in_collection / doc_freq)
                else:
                    idf = 0 # Term exists but has 0 doc freq? Should not happen with MinDF=2

                if idf == 0: # Skip terms present in all documents (or error case)
                    # print(f"  Skipping term '{term}' (IDF=0)")
                    continue

                # Fetch postings list (documents containing the term and their TF)
                cursor.execute("SELECT DocID, Term_Freq FROM Posting WHERE Term = %s", (term[:255],)) # Ensure term is truncated
                postings = cursor.fetchall()

                # Accumulate scores for documents
                for posting in postings:
                    doc_id = posting['DocID']
                    term_freq_in_doc = posting['Term_Freq']

                    # Calculate TF weight (using logarithmic scaling: 1 + log(tf))
                    tf_weight = 1 + math.log10(term_freq_in_doc) if term_freq_in_doc > 0 else 0

                    # Add TF-IDF score to the document's total score
                    # Score(d) = Sum over query terms t found in d: (1 + log10(tf(t,d))) * log10(N/df(t))
                    doc_scores[doc_id] += tf_weight * idf
            # else:
                # print(f"  Term '{term}' not found in dictionary.")

        cursor.close()
        conn.close()

    except pymysql.Error as e:
        print(f"Database error during search: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred during search: {e}")
        return []

    # 4. Rank documents by score
    # Sort by score (descending), then by DocID (ascending) as a tie-breaker
    sorted_docs = sorted(doc_scores.items(), key=lambda item: (item[1], item[0]), reverse=True)

    end_time = time.time()
    print(f"Search completed in {end_time - start_time:.4f} seconds.")

    # 5. Return top N results
    return sorted_docs[:top_n]

def display_results(results):
    """Formats and displays search results."""
    if not results:
        print("\nNo relevant documents found.")
        return

    print(f"\nTop {len(results)} most relevant documents:")
    print("-" * 60)
    for i, (doc_id, score) in enumerate(results, 1):
        # Here you could potentially fetch the article title from another table if you stored it
        print(f"{i}. Document ID: {doc_id}")
        print(f"   Relevance Score (TF-IDF based): {score:.4f}")
        print("-" * 30)

# --- Main Execution ---
if __name__ == "__main__":
    print("=" * 60)
    print(" Simple Wikipedia Search Engine")
    print("=" * 60)
    print("Uses TF-IDF weighting to rank documents.")

    while True:
        user_query = input("\nEnter your search query (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break
        if not user_query.strip():
            print("Please enter a query.")
            continue

        search_results = search_documents(user_query)
        display_results(search_results)

    print("\nExiting search engine.")