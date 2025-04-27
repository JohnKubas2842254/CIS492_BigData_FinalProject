import json
import spacy #type: ignore
import re
import os
from collections import defaultdict, Counter
import pymysql #type: ignore
import time

# --- spaCy GPU Setup ---
# Check if GPU is available and prefer it
try:
    spacy.require_gpu()
    print("GPU available, using GPU.")

    nlp = spacy.load("en_core_web_lg") # Faster, good accuracy
except Exception as e:
    print(f"GPU not available or CuPy/spaCy GPU support not installed ({e}). Using CPU.")
    # Fallback to a CPU model if GPU fails
    nlp = spacy.load("en_core_web_lg") # Large model for good performance on CPU
    # nlp = spacy.load("en_core_web_sm") # Smaller/faster model for CPU testing

# Increase max length if needed for very long documents
# nlp.max_length = 2000000 # Adjust as necessary

# --- Database Setup (Identical to NLTK version) ---
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "2842254K", # Replace with your password if different
    "database": "BigData_FinalProject_spaCy",
    "charset": "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor
}

try:
    db = pymysql.connect(**DB_CONFIG)
    cursor = db.cursor()

    # Initialize and use the correct database
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']}")
    cursor.execute(f"USE {DB_CONFIG['database']}")

    # Drop tables if they exist (for a clean run)
    cursor.execute("DROP TABLE IF EXISTS Posting")
    cursor.execute("DROP TABLE IF EXISTS Dictionary")
    cursor.execute("DROP TABLE IF EXISTS IndexMetadata")

    # Create MySQL tables (Identical schema)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Dictionary (
        Term VARCHAR(255) PRIMARY KEY,
        TotalDocsFreq INT,
        TotalCollectionFreq INT
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Posting (
        Term VARCHAR(255),
        DocID VARCHAR(255),
        Term_Freq INT,
        PRIMARY KEY (Term, DocID),
        FOREIGN KEY (Term) REFERENCES Dictionary(Term) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS IndexMetadata (
        KeyName VARCHAR(50) PRIMARY KEY,
        Value BIGINT
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """)
    print(f"Database '{DB_CONFIG['database']}' initialized successfully.")

except pymysql.Error as e:
    print(f"Database connection or setup failed: {e}")
    exit(1)
# --- End Database Setup ---


def nlp_pipeline_spacy(text):
    """
    Applies NLP preprocessing using spaCy:
    1. Processes text with spaCy model (tokenization, POS, lemmatization).
    2. Extracts lemmatized, non-stopword, alphabetic tokens (unigrams).
    3. Extracts lemmatized noun chunks as potential multi-word terms.
    Returns a dictionary containing processed tokens.
    """
    if not isinstance(text, str) or not text.strip():
        return {'tokens': [], 'pos_tags': []} # Return empty if input is invalid

    # Process text with spaCy
    # Disable components not strictly needed for this task to potentially speed up
    # We need 'tok2vec', 'tagger', 'attribute_ruler', 'lemmatizer' for lemmas and POS
    # We might need 'parser' for noun chunks depending on the model
    # NER ('ner') is not used here.
    try:
        # You might need to adjust disabled components based on the model used
        with nlp.select_pipes(enable=['tok2vec', 'tagger', 'attribute_ruler', 'lemmatizer', 'parser']):
             doc = nlp(text)
    except ValueError as e:
         print(f"Skipping document due to spaCy processing error: {e}")
         # Handle cases where text might exceed model's max length or other issues
         return {'tokens': [], 'pos_tags': []}


    final_tokens_for_index = []
    pos_tags_list = [] # Store simple (token, POS) tuples for debugging/example

    # Process individual tokens (Unigrams)
    for token in doc:
        # Basic filtering: not stopword, is alphabetic, length > 1
        if not token.is_stop and token.is_alpha and len(token.lemma_) > 1:
            final_tokens_for_index.append(token.lemma_)
        # Store POS tag for debugging output
        pos_tags_list.append((token.text, token.pos_))

    # Process Noun Chunks (as potential multi-word terms)
    # Noun chunks capture phrases like "machine learning", "New York City"
    for chunk in doc.noun_chunks:
        # Lemmatize the chunk, filter out stop words within the chunk, join with underscore
        lemmatized_chunk_parts = [token.lemma_ for token in chunk if not token.is_stop and token.is_alpha]
        if len(lemmatized_chunk_parts) > 1: # Only consider multi-word chunks
            phrase = "_".join(lemmatized_chunk_parts)
            # Ensure the phrase itself is not empty and has substantial content
            if phrase and len(phrase) > 1:
                 final_tokens_for_index.append(phrase)

    # Return unique tokens for the index and POS tags for debugging
    # Using list(set(...)) to remove duplicates that might arise from unigrams also being part of chunks
    return {
        'tokens': list(set(final_tokens_for_index)),
        'pos_tags': pos_tags_list # For the example printout
    }

# --- Helper Functions (Identical to NLTK version) ---
def load_json_data(filepath):
    """Loads JSON data from a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # Handle potential multi-line JSON objects within a list
            content = f.read()
            # Attempt to parse as a single JSON array/object first
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                # If fails, try parsing as newline-delimited JSON (NDJSON)
                print(f"Warning: Standard JSON parsing failed for {filepath}. Attempting NDJSON.")
                data = [json.loads(line) for line in content.strip().split('\n') if line.strip()]
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from {filepath} even as NDJSON: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred loading {filepath}: {e}")
        return None


def insert_into_database(postings, doc_freq, collection_freq, processed_doc_ids_in_batch, all_processed_doc_ids_count, is_final_batch=False):
    """
    Insert processed data into MySQL database.
    Modified to accept total doc count separately for the final metadata update.
    """
    global db, cursor # Ensure we are using the global db connection

    # Reconnect if connection is lost
    try:
        db.ping(reconnect=True)
    except pymysql.Error as e:
        print(f"Database connection lost, attempting to reconnect... Error: {e}")
        try:
            db = pymysql.connect(**DB_CONFIG)
            cursor = db.cursor()
            cursor.execute(f"USE {DB_CONFIG['database']}")
            print("Database reconnected successfully.")
        except pymysql.Error as e_reconnect:
            print(f"Database reconnection failed: {e_reconnect}")
            return False # Indicate failure

    try:
        # --- Metadata Update (Only on Final Call) ---
        if is_final_batch:
            print("  Updating final metadata in the database...")
            meta_sql = "INSERT INTO IndexMetadata (KeyName, Value) VALUES (%s, %s) ON DUPLICATE KEY UPDATE Value=VALUES(Value)"
            # Use the total count passed in for the final update
            cursor.execute(meta_sql, ('TotalDocuments', all_processed_doc_ids_count))
            db.commit()
            print(f"  Updated TotalDocuments = {all_processed_doc_ids_count} in IndexMetadata.")
            # No further processing needed in the final metadata-only call
            return True

        # --- Term Filtering (Applied per batch before insertion) ---
        # Note: Applying filtering per batch might slightly change results compared to global filtering at the end.
        # However, it's necessary for memory efficiency.
        min_doc_freq_threshold = 2
        terms_to_keep = {term for term, freq in doc_freq.items() if freq >= min_doc_freq_threshold}
        filtered_count = len(terms_to_keep)
        total_count = len(doc_freq)
        print(f"  Batch Filtering: Keeping {filtered_count} terms (removed {total_count - filtered_count}) based on batch doc freq >= {min_doc_freq_threshold}.")

        # --- Insert into Dictionary table ---
        dict_data = []
        if not terms_to_keep: # Skip if no terms left after filtering
             print("  No terms left after frequency filtering in this batch.")
        else:
            for term in terms_to_keep:
                truncated_term = term[:255]
                # Get frequencies from the batch's data
                batch_doc_freq = doc_freq.get(term, 0)
                batch_coll_freq = collection_freq.get(term, 0)
                if batch_doc_freq > 0: # Only insert if the term actually appeared in this batch
                    dict_data.append((truncated_term, batch_doc_freq, batch_coll_freq))

        if dict_data:
            dict_sql = """
                INSERT INTO Dictionary (Term, TotalDocsFreq, TotalCollectionFreq)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    TotalDocsFreq = TotalDocsFreq + VALUES(TotalDocsFreq),
                    TotalCollectionFreq = TotalCollectionFreq + VALUES(TotalCollectionFreq)
            """
            try:
                cursor.executemany(dict_sql, dict_data)
                db.commit()
                print(f"  Inserted/Updated {len(dict_data)} terms in Dictionary for this batch.")
            except pymysql.Error as e:
                print(f"Database Error during Dictionary insert: {e}")
                db.rollback()
                # Decide if you want to stop or continue processing other files
                return False # Indicate failure for this batch

        # --- Insert into Posting table ---
        posting_data = []
        if not terms_to_keep: # Skip if no terms left after filtering
             print("  No postings to insert as no terms met frequency threshold.")
        else:
            for term in terms_to_keep:
                # Check if the term exists in the batch's postings
                if term in postings:
                    truncated_term = term[:255]
                    for doc_id, freq in postings[term]:
                        # Ensure doc_id is one processed in this specific batch
                        if doc_id in processed_doc_ids_in_batch:
                             truncated_doc_id = doc_id[:255]
                             posting_data.append((truncated_term, truncated_doc_id, freq))

        if posting_data:
            batch_size = 10000 # Database batch insert size
            inserted_postings_count = 0
            posting_sql = """
                INSERT INTO Posting (Term, DocID, Term_Freq)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE Term_Freq = Term_Freq + VALUES(Term_Freq)
            """
            try:
                for i in range(0, len(posting_data), batch_size):
                    batch = posting_data[i:i + batch_size]
                    cursor.executemany(posting_sql, batch)
                    inserted_postings_count += len(batch)
                    # Commit frequently to release locks and manage transaction size
                    db.commit()
                    print(f"    Inserted posting batch {i//batch_size + 1}, total postings in this update: {inserted_postings_count}")
                print(f"  Inserted {inserted_postings_count} entries into Posting for this batch.")
            except pymysql.Error as e:
                 print(f"Database Error during Posting insert: {e}")
                 db.rollback()
                 # Decide if you want to stop or continue processing other files
                 return False # Indicate failure for this batch

        return True # Indicate success for this batch

    except pymysql.Error as e:
        print(f"General Database Error in insert_into_database: {e}")
        try:
            db.rollback()
        except pymysql.Error as rb_err:
            print(f"Rollback failed: {rb_err}")
        return False # Indicate failure
    except Exception as e:
        print(f"Unexpected Error in insert_into_database: {e}")
        return False # Indicate failure


# --- Main execution ---
if __name__ == "__main__":
    main_start_time = time.time()

    # Specify the directory containing JSON files
    json_directory = r'C:\Users\swoos\OneDrive\Documents\GitHub\CIS492_BigData_FinalProject\enwiki20201020' # Adjust if needed

    # Set for tracking all unique doc IDs across all files (for final count)
    all_processed_doc_ids_global = set()
    total_files_processed_successfully = 0
    total_articles_processed = 0

    print(f"Scanning directory: {json_directory}")
    try:
        json_files = [f for f in os.listdir(json_directory) if f.endswith('.json')]
        total_files_found = len(json_files)
        if total_files_found == 0:
            print("No JSON files found in the directory. Exiting.")
            exit()
        print(f"Found {total_files_found} JSON files to process.")
    except FileNotFoundError:
        print(f"Error: Directory not found: {json_directory}")
        exit(1)
    except Exception as e:
        print(f"Error scanning directory: {e}")
        exit(1)


    # Process each file individually and update database after each
    for i, json_filename in enumerate(json_files):
        file_start_time = time.time()
        # Reset in-memory structures for each file/batch
        postings_batch = defaultdict(list)
        doc_freq_batch = defaultdict(int)
        collection_freq_batch = defaultdict(int)
        file_processed_doc_ids_batch = set() # Track DocIDs processed in *this* file

        json_file_path = os.path.join(json_directory, json_filename)
        print(f"\n--- Processing file {i + 1}/{total_files_found}: {json_filename} ---")

        articles = load_json_data(json_file_path)
        load_end_time = time.time() # Time after loading JSON

        if articles and isinstance(articles, list):
            file_article_count = len(articles)
            print(f"Successfully loaded {len(articles)} articles from {json_filename} (Load time: {load_end_time - file_start_time:.2f}s)")
            
            print("Processing articles with spaCy using nlp.pipe...")
            processed_article_count_in_file = 0

            # --- Prepare data for nlp.pipe ---
            texts_to_process = []
            doc_ids_titles = [] # Keep track of corresponding IDs/titles
            valid_article_indices = [] # Keep track of original indices

            for article_index, article in enumerate(articles):
                if not isinstance(article, dict):
                    print(f"  Skipping item {article_index}: not a dictionary.")
                    continue
                doc_id = article.get('id')
                text = article.get('text')
                title = article.get('title', 'N/A')
                if not doc_id or not text:
                    print(f"  Skipping article (Index: {article_index}, Title: '{title}') due to missing ID or text.")
                    continue

                texts_to_process.append(text)
                doc_ids_titles.append({'id': doc_id, 'title': title})
                valid_article_indices.append(article_index) # Store original index

            num_articles_in_file = len(texts_to_process)

            # --- Process texts in batches using nlp.pipe ---
            # Adjust batch_size based on GPU memory and document complexity
            # Start with a moderate size like 128 or 256
            batch_size = 64 # Adjust as needed for your GPU memory
            nlp_start_time = time.time()
            nlp_docs = list(nlp.pipe(texts_to_process, batch_size=batch_size))
            nlp_end_time = time.time() # Time after nlp.pipe finishes
            print(f"  Finished nlp.pipe() (Time: {nlp_end_time - nlp_start_time:.2f}s)")

            del texts_to_process # Now safe to delete

            extraction_start_time = time.time()
            # --- Process results from nlp.pipe ---
            for idx, doc in enumerate(nlp_docs):
                doc_id = doc_ids_titles[idx]['id']
                title = doc_ids_titles[idx]['title']
                original_article_index = valid_article_indices[idx] # Get original index if needed

                # Add doc_id to the set for this file and the global set
                file_processed_doc_ids_batch.add(doc_id)
                all_processed_doc_ids_global.add(doc_id)

                # --- Extract tokens and POS tags from the processed doc ---
                # (This part moves out of nlp_pipeline_spacy function)
                final_tokens_for_index = []
                pos_tags_list = []

                for token in doc:
                    if not token.is_stop and token.is_alpha and len(token.lemma_) > 1:
                        final_tokens_for_index.append(token.lemma_)
                    pos_tags_list.append((token.text, token.pos_))

                for chunk in doc.noun_chunks:
                    lemmatized_chunk_parts = [token.lemma_ for token in chunk if not token.is_stop and token.is_alpha]
                    if len(lemmatized_chunk_parts) > 1:
                        phrase = "_".join(lemmatized_chunk_parts)
                        if phrase and len(phrase) > 1:
                            final_tokens_for_index.append(phrase)

                tokens_for_index = list(set(final_tokens_for_index))
                pos_tags_for_debug = pos_tags_list
                # --- End token extraction ---

                # Show example for the very first article of the first file only
                # Note: This condition needs adjustment as we process batches
                if i == 0 and idx == 0: # Check based on batch index now
                    print(f"\n--- spaCy POS Tags (first 30) for first article ('{title}') ---")
                    print(pos_tags_for_debug[:30])
                    print("-----------------------------------------------------------\n")
                    print(f"\n--- spaCy Tokens (first 50) for first article ('{title}') ---")
                    print(sorted(tokens_for_index)[:50])
                    print("-----------------------------------------------------------\n")

                if not tokens_for_index:
                    processed_article_count_in_file += 1
                    continue

                # Calculate term frequencies for this document
                term_counts = Counter(tokens_for_index)

                # Update the in-memory index structures for this batch
                processed_terms_in_doc = set()
                for term, freq in term_counts.items():
                    truncated_term = term[:255]
                    truncated_doc_id = doc_id[:255] # Truncate doc_id here

                    postings_batch[truncated_term].append((truncated_doc_id, freq))
                    collection_freq_batch[truncated_term] += freq
                    if truncated_term not in processed_terms_in_doc:
                        doc_freq_batch[truncated_term] += 1
                        processed_terms_in_doc.add(truncated_term)

                processed_article_count_in_file += 1
                total_articles_processed += 1
                # Progress indicator within the file (adjust frequency if needed)
                if processed_article_count_in_file % 500 == 0: # Maybe report less often with batching
                    print(f"  Processed {processed_article_count_in_file}/{num_articles_in_file} articles in batch...")
                    extraction_end_time = time.time() # Time after processing all docs from nlp.pipe
                    print(f"  Finished token extraction/aggregation (Time: {extraction_end_time - extraction_start_time:.2f}s)")

            extraction_end_time = time.time() # Time after processing all docs from nlp.pipe
            print(f"  Finished token extraction/aggregation (Time: {extraction_end_time - extraction_start_time:.2f}s)") # Print total extraction time here

            del nlp_docs # Free up memory after processing

            print(f"Finished processing {processed_article_count_in_file} articles from {json_filename}.")

            # After processing the file, insert its data into the database
            print(f"Inserting data from {json_filename} into database...")
            db_start_time = time.time()
            # Pass only the doc IDs processed in this specific batch for posting filtering
            # Pass 0 for total doc count as this is not the final batch
            insert_success = insert_into_database(
                postings_batch,
                doc_freq_batch,
                collection_freq_batch,
                file_processed_doc_ids_batch, # Doc IDs from this file only
                0, # Placeholder for total count, not used here
                is_final_batch=False
            )

            db_end_time = time.time() # Time after database insert finishes
            print(f"  Finished database insertion (Time: {db_end_time - db_start_time:.2f}s)")
            # --- End timing for database insert ---

            if insert_success:
                total_files_processed_successfully += 1
                print(f"Successfully updated database with data from {json_filename}")
            else:
                print(f"Failed to update database completely for {json_filename}. Check errors above.")
                # Optional: Decide whether to stop processing or continue with next file

        elif articles is None:
            print(f"Failed to load or decode articles from {json_filename}. Skipping file.")
        else:
            print(f"Loaded data from {json_filename} is not a list as expected (Type: {type(articles)}). Skipping file.")

        file_end_time = time.time()
        print(f"Time taken for {json_filename}: {file_end_time - file_start_time:.2f} seconds")


    # --- Final Metadata Update ---
    print("\n--- Performing final metadata update ---")
    # Pass the final count of all unique documents processed across all files
    final_update_success = insert_into_database({}, {}, {}, set(), len(all_processed_doc_ids_global), is_final_batch=True)
    if not final_update_success:
         print("Error during final metadata update.")


    # --- Processing Summary ---
    print("\n--- Total Processing Summary ---")
    print(f"Successfully processed and attempted DB insert for {total_files_processed_successfully}/{total_files_found} JSON files.")
    print(f"Total articles processed across all files: {total_articles_processed}")
    final_doc_count = len(all_processed_doc_ids_global)
    print(f"Total unique documents processed (N): {final_doc_count}")


    # --- Close Database Connection ---
    if 'db' in locals() and db.open:
        try:
            cursor.close()
            db.close()
            print("Database connection closed.")
        except pymysql.Error as e:
            print(f"Error closing database connection: {e}")

    main_end_time = time.time()
    print(f"\nTotal execution time: {main_end_time - main_start_time:.2f} seconds")
