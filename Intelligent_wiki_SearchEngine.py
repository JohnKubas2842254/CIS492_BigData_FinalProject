import json
import spacy #type: ignore
import re
import os # Import os if not already present
from collections import defaultdict, Counter
import pymysql #type: ignore
import time
import math

# --- spaCy Model Loading ---
# Use the same model and setup as your indexer
try:
    nlp = spacy.load("en_core_web_lg")
    print("spaCy model 'en_core_web_lg' loaded successfully.")
except Exception as e:
    print(f"Could not load spaCy model 'en_core_web_lg': {e}")
    print("Ensure the model is downloaded: python -m spacy download en_core_web_lg")
    exit(1)
# --- End spaCy Model Loading ---


# --- Database Configuration ---
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "2842254K",
    "database": "BigData_FinalProject_spaCy",
    "charset": "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor
}

# --- spaCy NLP Pipeline (Copied from JSON_to_InvertedIndex_spaCy.py) ---
# Note: This processes the QUERY, not documents during search.
# It should mirror the *token extraction* logic used during indexing.
def nlp_pipeline_spacy_query(text):
    """
    Applies NLP preprocessing using spaCy specifically for QUERY processing.
    Mirrors the token extraction logic from the indexing script.
    """
    if not isinstance(text, str) or not text.strip():
        return {'tokens': []}

    try:
        # Process query text
        with nlp.select_pipes(enable=['tok2vec', 'tagger', 'attribute_ruler', 'lemmatizer', 'parser']):
             doc = nlp(text)
    except ValueError as e:
         print(f"Skipping query processing due to spaCy error: {e}")
         return {'tokens': []}

    final_query_tokens = []

    # Process individual tokens (Unigrams)
    for token in doc:
        if not token.is_stop and token.is_alpha and len(token.lemma_) > 1:
            final_query_tokens.append(token.lemma_)

    # Process Noun Chunks
    for chunk in doc.noun_chunks:
        lemmatized_chunk_parts = [token.lemma_ for token in chunk if not token.is_stop and token.is_alpha]
        if len(lemmatized_chunk_parts) > 1:
            phrase = "_".join(lemmatized_chunk_parts)
            if phrase and len(phrase) > 1:
                 final_query_tokens.append(phrase)

    # Return unique tokens for the query
    # We need counts for query TF, so don't use set() here yet.
    # Counter will handle uniqueness and counts later.
    return {'tokens': final_query_tokens} # Return list with potential duplicates
# --- End of spaCy NLP Pipeline ---

def search_documents(query_text, top_n=10):
    """
    Processes a query, calculates Dot Product scores, normalizes by RawWordCount,
    and returns top N documents.
    """
    start_time = time.time()

    # 1. Process Query & Calculate Query TF-IDF Weights
    nlp_result = nlp_pipeline_spacy_query(query_text)
    query_term_counts = Counter(nlp_result['tokens'])

    if not query_term_counts:
        print("Query could not be processed or resulted in no valid terms.")
        return []

    processed_query_terms = list(query_term_counts.keys())
    print(f"Processed query terms: {processed_query_terms}")

    doc_scores_dot_product = defaultdict(float) # Stores dot product scores
    query_tfidf_weights = {}
    query_vector_magnitude_sq = 0.0 # Not used for ranking, but calculated

    total_docs_in_collection = 0
    conn = None
    cursor = None

    try:
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # 2. Get N (Total Documents)
        # ... (same N retrieval logic with fallback as before) ...
        cursor.execute("SELECT Value FROM IndexMetadata WHERE KeyName = 'TotalDocuments'")
        result = cursor.fetchone()
        if result and result['Value'] > 0:
            total_docs_in_collection = result['Value']
            print(f"Retrieved TotalDocuments (N) = {total_docs_in_collection}")
        else:
            print("Error: Could not retrieve TotalDocuments count from IndexMetadata. Attempting fallback...")
            try:
                cursor.execute("SELECT COUNT(DISTINCT DocID) as count FROM Posting")
                result_fallback = cursor.fetchone()
                if result_fallback and result_fallback['count'] > 0:
                    total_docs_in_collection = result_fallback['count']
                    print(f"Fallback successful: N = {total_docs_in_collection}")
                else:
                     raise ValueError("Fallback count failed.")
            except Exception as e_fallback:
                 print(f"Fallback count query failed: {e_fallback}. Aborting.")
                 raise

        if total_docs_in_collection <= 1:
             print("Warning: Total document count is very low.")

        # --- Calculate Query TF-IDF weights & IDFs ---
        print("Calculating query weights and term IDFs...")
        term_idfs = {} # Store IDFs
        for term, query_tf_raw in query_term_counts.items():
            cursor.execute("SELECT Term, TotalDocsFreq FROM Dictionary WHERE Term = %s", (term[:255],))
            dict_entry = cursor.fetchone()
            idf = 0
            if dict_entry:
                doc_freq = dict_entry['TotalDocsFreq']
                if doc_freq > 0 and total_docs_in_collection >= doc_freq:
                     idf = math.log10(total_docs_in_collection / doc_freq)
            if idf > 0:
                term_idfs[term] = idf
                query_tf_weight = 1 + math.log10(query_tf_raw)
                query_tfidf = query_tf_weight * idf
                query_tfidf_weights[term] = query_tfidf
                query_vector_magnitude_sq += query_tfidf ** 2

        if not query_tfidf_weights:
             print("None of the query terms were found in the document collection or IDF was zero for all.")
             return []

        query_magnitude = math.sqrt(query_vector_magnitude_sq) if query_vector_magnitude_sq > 0 else 1.0

        # 3. Calculate Dot Product scores for documents
        print("Calculating initial dot products...")
        processed_terms_count = 0
        candidate_doc_ids = set()
        for term, query_tfidf in query_tfidf_weights.items():
            processed_terms_count += 1
            # print(f"  Processing term {processed_terms_count}/{len(query_tfidf_weights)}: '{term}'") # Optional verbose print
            idf = term_idfs[term]
            cursor.execute("SELECT DocID, Term_Freq FROM Posting WHERE Term = %s", (term[:255],))
            postings = cursor.fetchall()
            for posting in postings:
                doc_id = posting['DocID']
                term_freq_in_doc = posting['Term_Freq']
                doc_tf_weight = 1 + math.log10(term_freq_in_doc) if term_freq_in_doc > 0 else 0
                doc_tfidf = doc_tf_weight * idf
                doc_scores_dot_product[doc_id] += query_tfidf * doc_tfidf
                candidate_doc_ids.add(doc_id)

        if not candidate_doc_ids:
            print("No documents found containing any of the query terms.")
            return []

        print(f"Found {len(candidate_doc_ids)} candidate documents.")

        # 4. Fetch RawWordCount for candidate documents
        print("Fetching document lengths for normalization...")
        doc_lengths = {}
        candidate_list = list(candidate_doc_ids)
        # Fetch lengths in batches to avoid huge IN clauses
        fetch_batch_size = 1000
        for i in range(0, len(candidate_list), fetch_batch_size):
            batch_ids = candidate_list[i:i+fetch_batch_size]
            format_strings = ','.join(['%s'] * len(batch_ids))
            sql = f"SELECT DocID, RawWordCount FROM DocumentMetadata WHERE DocID IN ({format_strings})"
            cursor.execute(sql, tuple(batch_ids))
            for row in cursor.fetchall():
                # Use max(1, count) to avoid division by zero for empty docs
                doc_lengths[row['DocID']] = max(1, row.get('RawWordCount', 1))

        print(f"Fetched lengths for {len(doc_lengths)} documents.")

        # 5. Calculate Final Normalized Scores
        final_scores = {}
        for doc_id, dot_product in doc_scores_dot_product.items():
            length = doc_lengths.get(doc_id, 1) # Default to 1 if length not found
            final_scores[doc_id] = dot_product / length # Normalize by raw word count

        # 6. Rank documents by the normalized score
        sorted_docs = sorted(final_scores.items(), key=lambda item: item[1], reverse=True)

        end_time = time.time()
        print(f"Search completed in {end_time - start_time:.4f} seconds.")

        # 7. Return top N results
        return sorted_docs[:top_n]

    except Exception as e:
        print(f"An error occurred during search: {e}")
        # Ensure connection is closed if open
        if conn and conn.open:
            if cursor: cursor.close()
            conn.close()
        return [] # Return empty list on error
    finally:
        # Ensure connection is always closed
        if conn and conn.open:
            if cursor: cursor.close()
            conn.close()

def display_results(results):
    """Formats and displays search results."""
    if not results:
        print("\nNo relevant documents found.")
        return

    print(f"\nTop {len(results)} most relevant documents:")
    print("-" * 60)
    for i, (doc_id, score) in enumerate(results, 1):
        # Fetch the article title from the original JSON data if possible/needed
        # This requires loading/accessing the JSON data again, which might be slow
        # or storing titles separately during indexing.
        # For now, just display the ID and score.
        print(f"{i}. Document ID: {doc_id}")
        # Score is now the Dot Product, not a normalized Cosine Similarity value
        print(f"   Relevance Score (Dot Product): {score:.4f}")
        print("-" * 30)


# --- Main Execution ---
if __name__ == "__main__":
    print("=" * 60)
    print(" Simple Wikipedia Search Engine (spaCy + Dot Product Ranking)")
    print("=" * 60)

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