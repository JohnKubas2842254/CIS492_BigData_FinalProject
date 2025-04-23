import json
import nltk #type: ignore
import re
from nltk.corpus import stopwords #type: ignore
from nltk.stem import WordNetLemmatizer #type: ignore
from nltk.util import ngrams #type: ignore
from collections import defaultdict, Counter
import pymysql #type: ignore
import time

# --- NLTK Data Download (Uncomment if needed) ---
# try:
#     nltk.data.find('corpora/wordnet')
# except nltk.downloader.DownloadError:
#     nltk.download('wordnet')
# try:
#     nltk.data.find('corpora/stopwords')
# except nltk.downloader.DownloadError:
#     nltk.download('stopwords')
# try:
#     nltk.data.find('tokenizers/punkt')
# except nltk.downloader.DownloadError:
#     nltk.download('punkt')
# try:
#     nltk.data.find('taggers/averaged_perceptron_tagger')
# except nltk.downloader.DownloadError:
#     nltk.download('averaged_perceptron_tagger')
# --- End NLTK Data Download ---

# Initialize lemmatizer and stopwords list
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# MySQL connection setup
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "2842254K",
    "database": "BigData_FinalProject",
    "charset": "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor
}

db = pymysql.connect(**DB_CONFIG)
cursor = db.cursor()

#initialize and use the correct database
cursor.execute("CREATE DATABASE IF NOT EXISTS BigData_FinalProject")
cursor.execute("USE BigData_FinalProject")

cursor.execute("DROP TABLE IF EXISTS Posting")
cursor.execute("DROP TABLE IF EXISTS Dictionary")

# Create MySQL tables
cursor.execute("""
CREATE TABLE IF NOT EXISTS Dictionary (
    Term VARCHAR(255) PRIMARY KEY,
    TotalDocsFreq INT,
    TotalCollectionFreq INT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin; 
""") # Added ENGINE and COLLATE for better indexing/case sensitivity if needed

cursor.execute("""
CREATE TABLE IF NOT EXISTS Posting (
    Term VARCHAR(255),
    DocID VARCHAR(255),
    Term_Freq INT,
    PRIMARY KEY (Term, DocID),
    FOREIGN KEY (Term) REFERENCES Dictionary(Term) ON DELETE CASCADE 
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;
""") # Added Foreign Key and ENGINE/COLLATE

def nlp_pipeline(text):
    """
    Applies NLP preprocessing steps including n-gram extraction:
    1. Lowercasing
    2. Punctuation removal (keeps numbers)
    3. Tokenization
    4. POS Tagging
    5. Unigram processing (stop word removal, lemmatization, alpha filtering)
    6. N-gram generation (bi-grams, tri-grams)
    7. N-gram filtering based on POS patterns
    8. N-gram processing (lemmatization)
    9. Combine unigrams and processed n-grams
    Returns a dictionary containing processed tokens and POS tags.
    """
    if not isinstance(text, str):
        return {'tokens': [], 'pos_tags': []}
    
    # 1. Lowercasing
    text_lower = text.lower()

    # 2. Punctuation/Number removal (keeping only letters and spaces)
    text_cleaned = re.sub(r'[^a-z\s]', '', text_lower)

    # 3. Tokenization
    tokens = nltk.word_tokenize(text_cleaned)

    if not tokens:
        return {'tokens': [], 'pos_tags': [], 'ner_tree': None}

    # 4. POS Tagging
    pos_tags = nltk.pos_tag(tokens)

    final_tokens_for_index = []

    # 5. Unigram processing
    for word, tag in pos_tags:
        if word in stop_words:
            continue
        # Keep only alphabetic unigrams after lemmatization (consistent with previous approach)
        lemma = lemmatizer.lemmatize(word)
        if lemma.isalpha() and len(lemma) > 1:
            final_tokens_for_index.append(lemma)

    # 6. N-gram generation and 7. Filtering / 8. Processing
    # Define simple POS patterns for potentially meaningful n-grams
    # NNP(S) = Proper Noun, NN(S)=Noun, JJ=Adjective, CD=Cardinal Number
    allowed_bigram_patterns = [
        ('JJ', 'NN'), ('JJ', 'NNS'),
        ('NN', 'NN'), ('NNS', 'NN'), ('NN', 'NNS'), ('NNS', 'NNS'),
        ('NNP', 'NNP'), ('NNPS', 'NNP'), ('NNP', 'NNPS'), ('NNPS', 'NNPS'),
        ('NNP', 'CD'), # e.g., "Superbowl 52"
        ('NN', 'CD')   # e.g., "Version 3"
    ]
    allowed_trigram_patterns = [
        ('JJ', 'NN', 'NN'), ('JJ', 'NNS', 'NN'), ('JJ', 'NN', 'NNS'),
        ('NN', 'NN', 'NN'), ('NNS', 'NN', 'NN'), ('NN', 'NNS', 'NN'), ('NN', 'NN', 'NNS'),
        ('NNP', 'NNP', 'NNP'), ('NNP', 'NNP', 'CD'), # e.g., "Super Bowl 52"
        ('NNP', 'NN', 'NN'), # Proper noun followed by common nouns
        ('NN', 'IN', 'NN') # Noun-preposition-noun (e.g., "state of art") - less common, example
    ]

    # Process Bi-grams
    for gram in ngrams(pos_tags, 2):
        tags = tuple(tag for word, tag in gram)
        # Check prefix to handle NNP/NNPS, NN/NNS variations simply
        tag_prefixes = tuple(t[:2] for t in tags) # ('NN', 'NN'), ('JJ', 'NN'), ('NN', 'CD') etc.

        if tag_prefixes in allowed_bigram_patterns:
            words = [word for word, tag in gram]
            # Lemmatize words in the n-gram
            lemmatized_words = [lemmatizer.lemmatize(w) for w in words]
            # Basic validation: ensure words are not empty after potential lemmatization issues
            if all(lemmatized_words):
                 # Join with underscore to form a single term
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

    # 9. Return combined list (unigrams + n-grams)
    return {
        # Use set to remove duplicates if unigram processing added something also in an n-gram
        'tokens': final_tokens_for_index,
        'pos_tags': pos_tags
    }

def load_json_data(filepath):
    """Loads JSON data from a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}")
        return None

# --- Placeholder for NLP Pipeline Functions ---
# We will add functions for tokenization, stop word removal, etc. here later.

# --- Main execution ---
if __name__ == "__main__":
    start_time = time.time()
    json_file_path = '/Users/kube/VSprojects/CIS492_BigData_FinalProject/enwiki20201020/0b8a29b0-177a-4103-b124-18a145f0a564.json'
    articles = load_json_data(json_file_path)

    if articles:
        print(f"Successfully loaded {len(articles)} articles from {json_file_path}")

        # In-memory structures for the inverted index
        postings = defaultdict(list) # Term -> [(DocID, Term_Freq), ...]
        doc_freq = defaultdict(int)       # Term -> TotalDocsFreq
        collection_freq = defaultdict(int) # Term -> TotalCollectionFreq

        print("Processing articles...")
        processed_count = 0
        for article in articles:
            doc_id = article.get('id')
            text = article.get('text')
            title = article.get('title', 'N/A')

            if not doc_id or not text:
                print(f"Skipping article due to missing ID or text: {article.get('title', 'N/A')}")
                continue

            # Process text using the NLP pipeline
            nlp_results = nlp_pipeline(text)
            tokens_for_index = nlp_results['tokens'] # Get the list of words for indexing
            pos_tags = nlp_results['pos_tags']
            # ner_tree = nlp_results['ner_tree']

            # # --- Example: Print NER results for the first article ---
            # if processed_count == 0 and nlp_results['ner_tree']:
            #      print(f"\n--- NER Tree for first article () ---")
            #      print("Named Entities Found:")
            #      for chunk in nlp_results['ner_tree']:
            #          if hasattr(chunk, 'label'):
            #              entity_label = chunk.label()
            #              entity_text = ' '.join(c[0] for c in chunk.leaves())
            #              print(f'  {entity_label}: {entity_text}')
            #      print("-------------------------------------------\n")
            # # --- End Example ---


            # --- Example: Print POS tags and Tokens for the first article ---
            if processed_count == 0:
                 print(f"\n--- POS Tags for first article ('{title}') ---")
                 print(pos_tags[:30])
                 print("-------------------------------------------\n")
                 print(f"\n--- Tokens (Unigrams+Ngrams) for first article ('{title}') ---")
                 # Print first ~50 tokens as example
                 print(sorted(tokens_for_index)[:50]) # Sort for easier viewing
                 print("-------------------------------------------\n")
            # --- End Example ---

            if not tokens_for_index: # Skip if no tokens resulted after processing
                continue

            # Calculate term frequencies for the current document
            term_counts = Counter(tokens_for_index)

            # Update the in-memory index
            for term, freq in term_counts.items():
                postings[term].append((doc_id, freq))
                doc_freq[term] += 1 # Increment document frequency for the term
                collection_freq[term] += freq # Add to total collection frequency

            processed_count += 1
            if processed_count % 100 == 0: # Print progress update
                 print(f"  Processed {processed_count}/{len(articles)} articles...")

        #print(f"Finished processing {processed_count} articles.")
        initial_term_count = len(postings)
        print(f"Found {initial_term_count} unique terms initially.")

        # --- Minimum Document Frequency Filtering ---
        min_doc_freq_threshold = 2
        print(f"Applying Minimum Document Frequency filter (threshold = {min_doc_freq_threshold})...")

        # Create new dictionaries to hold the filtered data
        filtered_postings = {}
        filtered_doc_freq = {}
        filtered_collection_freq = {}

        terms_to_keep = {term for term, freq in doc_freq.items() if freq >= min_doc_freq_threshold}

        # Populate filtered dictionaries
        for term in terms_to_keep:
            filtered_postings[term] = postings[term]
            filtered_doc_freq[term] = doc_freq[term]
            filtered_collection_freq[term] = collection_freq[term]

        filtered_term_count = len(filtered_postings)
        print(f"Kept {filtered_term_count} terms after filtering (removed {initial_term_count - filtered_term_count} terms).")

        # --- Database Insertion ---
        print("Inserting filtered data into MySQL...")
        try:
            # Clear existing tables
            print("  Clearing existing Dictionary and Posting tables...")
            cursor.execute("DELETE FROM Posting")
            cursor.execute("DELETE FROM Dictionary")
            db.commit()
            print("  Tables cleared.")

            # Insert into Dictionary table
            dict_data = []
            for term in filtered_postings.keys():
                 # Ensure term length doesn't exceed VARCHAR(255)
                 truncated_term = term[:255] 
                 dict_data.append((truncated_term, doc_freq[term], collection_freq[term]))

            dict_sql = "INSERT INTO Dictionary (Term, TotalDocsFreq, TotalCollectionFreq) VALUES (%s, %s, %s)"
            cursor.executemany(dict_sql, dict_data)
            db.commit()
            print(f"  Inserted {len(dict_data)} terms into Dictionary.")
            #print(f"  Inserted/Updated {len(dict_data)} terms in Dictionary.")

            # Insert into Posting table
            posting_data = []
            for term, doc_list in filtered_postings.items():
                truncated_term = term[:255] # Use the same truncated term
                for doc_id, freq in doc_list:
                    # Ensure doc_id length doesn't exceed VARCHAR(255)
                    truncated_doc_id = doc_id[:255] 
                    posting_data.append((truncated_term, truncated_doc_id, freq))

            # Consider batching if posting_data is very large
            batch_size = 10000 
            posting_sql = "INSERT INTO Posting (Term, DocID, Term_Freq) VALUES (%s, %s, %s) ON DUPLICATE KEY UPDATE Term_Freq=VALUES(Term_Freq)" # Or handle duplicates as needed
            
            inserted_postings = 0
            for i in range(0, len(posting_data), batch_size):
                batch = posting_data[i:i + batch_size]
                rows_affected = cursor.executemany(posting_sql, batch)
                inserted_postings += cursor.executemany(posting_sql, batch)
                db.commit() # Commit each batch
                print(f"    Inserted batch {i//batch_size + 1}, total postings: {inserted_postings}")

            print(f"  Inserted {inserted_postings} entries into Posting.")

            # Final commit (may not be strictly necessary if committing batches)
            db.commit()
            print("Database insertion complete.")

        except pymysql.Error as e:
            print(f"Database Error: {e}")
            db.rollback()
        finally:
            if 'db' in locals() and db.open:
                 cursor.close()
                 db.close()
                 print("Database connection closed.")
    else:
        print("Failed to load articles.")

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
