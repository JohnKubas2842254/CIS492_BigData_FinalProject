import json
import nltk #type: ignore
import re
import os
from nltk.corpus import stopwords #type: ignore
from nltk.stem import WordNetLemmatizer #type: ignore
from nltk.util import ngrams #type: ignore
from collections import defaultdict, Counter
import pymysql #type: ignore
import time

# --- NLTK Data Download (Uncomment if needed) ---
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng')
try: 
    nltk.data.find('punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
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
cursor.execute("DROP TABLE IF EXISTS IndexMetadata")

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

cursor.execute("""
CREATE TABLE IF NOT EXISTS IndexMetadata (
    KeyName VARCHAR(50) PRIMARY KEY,
    Value BIGINT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
""")

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

def insert_into_database(postings, doc_freq, collection_freq, all_processed_doc_ids, is_final_batch=False):
    """
    Insert processed data into MySQL database
    
    Args:
        postings: Dictionary mapping terms to document frequency lists
        doc_freq: Dictionary of document frequencies for each term
        collection_freq: Dictionary of collection frequencies for each term
        all_processed_doc_ids: Set of all document IDs processed
        is_final_batch: Boolean indicating if this is the final batch
    
    Returns:
        Tuple of (success_status, processed_docs_count)
    """
    try:
        # Only clear tables on the first run
        if is_final_batch:
            print("  Updating metadata in the database...")
            # Update total document count in IndexMetadata
            meta_sql = "INSERT INTO IndexMetadata (KeyName, Value) VALUES (%s, %s) ON DUPLICATE KEY UPDATE Value=VALUES(Value)"
            cursor.execute(meta_sql, ('TotalDocuments', len(all_processed_doc_ids)))
            db.commit()
            print(f"  Updated TotalDocuments = {len(all_processed_doc_ids)} in IndexMetadata.")
        
        # Apply minimum document frequency filtering
        min_doc_freq_threshold = 2
        if not is_final_batch:
            print(f"Applying Minimum Document Frequency filter (threshold = {min_doc_freq_threshold})...")
        
        # Filter terms
        terms_to_keep = {term for term, freq in doc_freq.items() if freq >= min_doc_freq_threshold}
        filtered_count = len(terms_to_keep)
        total_count = len(doc_freq)
        if not is_final_batch:
            print(f"Keeping {filtered_count} terms after filtering (removed {total_count - filtered_count} terms).")
        
        # Insert into Dictionary table
        dict_data = []
        for term in terms_to_keep:
            # Ensure term length doesn't exceed VARCHAR(255)
            truncated_term = term[:255] 
            dict_data.append((truncated_term, doc_freq[term], collection_freq[term]))
        
        if dict_data:
            dict_sql = """
                INSERT INTO Dictionary (Term, TotalDocsFreq, TotalCollectionFreq) 
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE 
                    TotalDocsFreq = TotalDocsFreq + VALUES(TotalDocsFreq),
                    TotalCollectionFreq = TotalCollectionFreq + VALUES(TotalCollectionFreq)
            """
            cursor.executemany(dict_sql, dict_data)
            db.commit()
            print(f"  Inserted/Updated {len(dict_data)} terms in Dictionary.")
        
        # Insert into Posting table
        posting_data = []
        for term in terms_to_keep:
            if term in postings:
                truncated_term = term[:255]  # Use the same truncated term
                for doc_id, freq in postings[term]:
                    # Ensure doc_id length doesn't exceed VARCHAR(255)
                    truncated_doc_id = doc_id[:255] 
                    posting_data.append((truncated_term, truncated_doc_id, freq))
        
        # Consider batching for posting data
        batch_size = 10000
        inserted_postings = 0
        
        if posting_data:
            posting_sql = """
                INSERT INTO Posting (Term, DocID, Term_Freq) 
                VALUES (%s, %s, %s) 
                ON DUPLICATE KEY UPDATE Term_Freq = Term_Freq + VALUES(Term_Freq)
            """
            
            for i in range(0, len(posting_data), batch_size):
                batch = posting_data[i:i + batch_size]
                cursor.executemany(posting_sql, batch)
                inserted_postings += len(batch)
                db.commit()  # Commit each batch
                print(f"    Inserted batch {i//batch_size + 1}, total postings in this update: {inserted_postings}")
        
            print(f"  Inserted {inserted_postings} entries into Posting.")
        
        return True, len(all_processed_doc_ids)
    
    except pymysql.Error as e:
        print(f"Database Error: {e}")
        db.rollback()
        return False, 0

# --- Main execution ---
if __name__ == "__main__":
    start_time = time.time()
    
    # Specify the directory containing JSON files
    json_directory = r'C:\Users\swoos\OneDrive\Documents\GitHub\CIS492_BigData_FinalProject\enwiki20201020'
    
    # Clear existing tables once at the beginning
    try:
        print("Initializing database tables...")
        cursor.execute("DELETE FROM Posting")
        cursor.execute("DELETE FROM Dictionary")
        cursor.execute("DELETE FROM IndexMetadata")
        db.commit()
        print("Tables cleared and ready for new data.")
    except pymysql.Error as e:
        print(f"Database Error during initialization: {e}")
        if 'db' in locals() and db.open:
            db.rollback()
            cursor.close()
            db.close()
        exit(1)
    
    # Set for tracking all unique doc IDs across all files
    all_processed_doc_ids = set()
    total_files = 0
    processed_files = 0
    total_articles = 0
    
    print(f"Scanning directory: {json_directory}")
    json_files = [f for f in os.listdir(json_directory) if f.endswith('.json')]
    total_files = len(json_files)
    print(f"Found {total_files} JSON files to process.")
    
    # Process each file individually and update database after each
    for json_filename in json_files:
        # Reset in-memory structures for each file
        postings = defaultdict(list)
        doc_freq = defaultdict(int)
        collection_freq = defaultdict(int)
        file_processed_doc_ids = set()
        
        json_file_path = os.path.join(json_directory, json_filename)
        print(f"\nProcessing file {processed_files + 1}/{total_files}: {json_filename}")
        
        articles = load_json_data(json_file_path)
        
        if articles:
            print(f"Successfully loaded {len(articles)} articles from {json_filename}")
            file_article_count = len(articles)
            total_articles += file_article_count
            
            print("Processing articles...")
            processed_count = 0
            for article in articles:
                doc_id = article.get('id')
                text = article.get('text')
                title = article.get('title', 'N/A')
                
                if not doc_id or not text:
                    print(f"Skipping article due to missing ID or text: {article.get('title', 'N/A')}")
                    continue
                
                file_processed_doc_ids.add(doc_id)
                all_processed_doc_ids.add(doc_id)
                
                # Process text using the NLP pipeline
                nlp_results = nlp_pipeline(text)
                tokens_for_index = nlp_results['tokens']
                pos_tags = nlp_results['pos_tags']
                
                # Show example for the very first article only
                if processed_files == 0 and processed_count == 0:
                    print(f"\n--- POS Tags for first article ('{title}') ---")
                    print(pos_tags[:30])
                    print("-------------------------------------------\n")
                    print(f"\n--- Tokens (Unigrams+Ngrams) for first article ('{title}') ---")
                    print(sorted(tokens_for_index)[:50])
                    print("-------------------------------------------\n")
                
                if not tokens_for_index:
                    processed_count += 1
                    continue
                
                # Calculate term frequencies for this document
                term_counts = Counter(tokens_for_index)
                
                # Update the in-memory index for this file
                for term, freq in term_counts.items():
                    postings[term].append((doc_id, freq))
                    doc_freq[term] += 1
                    collection_freq[term] += freq
                
                processed_count += 1
                if processed_count % 100 == 0:
                    print(f"  Processed {processed_count}/{file_article_count} articles...")
            
            print(f"Finished processing {processed_count} articles from {json_filename}")
            
            # After processing the file, insert data into the database
            print(f"Inserting data from {json_filename} into database...")
            success, _ = insert_into_database(postings, doc_freq, collection_freq, file_processed_doc_ids)
            
            if success:
                processed_files += 1
                print(f"Successfully updated database with data from {json_filename}")
            else:
                print(f"Failed to update database with data from {json_filename}")
        else:
            print(f"Failed to load articles from {json_filename}")
    
    # Final update to ensure metadata is correct
    print("\nPerforming final database update...")
    insert_into_database({}, {}, {}, all_processed_doc_ids, is_final_batch=True)
    
    # Processing summary
    print("\n--- Processing Summary ---")
    print(f"Processed {processed_files}/{total_files} JSON files.")
    print(f"Total articles processed: {total_articles}")
    print(f"Total unique documents processed (N): {len(all_processed_doc_ids)}")
    
    # Close database connection
    if 'db' in locals() and db.open:
        cursor.close()
        db.close()
        print("Database connection closed.")
    
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")