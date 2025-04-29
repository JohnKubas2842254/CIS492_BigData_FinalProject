import json
import spacy #type: ignore
import re
import os
from collections import defaultdict, Counter
import pymysql #type: ignore
import time
import csv
import threading
from queue import Queue

# --- spaCy GPU Setup ---
# Check if GPU is available and prefer it
try:
    spacy.require_gpu()
    print("GPU available, using GPU.")

    #nlp = spacy.load("en_core_web_lg")
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    print(f"GPU not available or CuPy/spaCy GPU support not installed ({e}). Using CPU.")
    # Fallback to a CPU model if GPU fails
    #nlp = spacy.load("en_core_web_lg")
    nlp = spacy.load("en_core_web_sm") # Smaller/faster model for CPU testing

# --- Configuration ---
MAX_TEXT_LENGTH = 1000000 # Max characters per article for NLP (spaCy default is often 1M)
# --- Batch Size Configuration ---
TARGET_NLP_BATCH_SIZE = 128 # Attempt this size first
FALLBACK_NLP_BATCH_SIZE = 64  # Fall back to this size on memory error
# --- End Batch Size Configuration ---
DB_POSTING_BATCH_SIZE = 250000
DB_META_BATCH_SIZE = 5000
DB_WORKER_COUNT = 3  # Number of parallel database workers
DB_COMMIT_INTERVAL = 50  # How many batches before committing
# --- End Configuration ---

# --- Database Setup ---
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "2842254K", 
    "database": "BigData_FinalProject_spaCy2",
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
    #cursor.execute("DROP TABLE IF EXISTS Posting")
    #cursor.execute("DROP TABLE IF EXISTS Dictionary")
    #cursor.execute("DROP TABLE IF EXISTS IndexMetadata")

    # Create MySQL tables
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
        CREATE TABLE IF NOT EXISTS DocumentMetadata (
            DocID VARCHAR(255) PRIMARY KEY,
            RawWordCount INT,
            MagnitudeSq DOUBLE NULL
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
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

# --- Load Custom Stopwords ---
CUSTOM_STOPWORDS_FILE = r'C:\Users\swoos\OneDrive\Documents\GitHub\CIS492_BigData_FinalProject\common_terms2.csv'
custom_stopwords = set()
try:
    with open(CUSTOM_STOPWORDS_FILE, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader) # Skip header row (TERM,TOTALCOLLECTIONFREQ)
        for row in reader:
            if row: # Ensure row is not empty
                custom_stopwords.add(row[0]) # Add the term from the first column
    print(f"Loaded {len(custom_stopwords)} custom stopwords from {CUSTOM_STOPWORDS_FILE}")
except FileNotFoundError:
    print(f"Warning: Custom stopwords file not found at {CUSTOM_STOPWORDS_FILE}. Proceeding without custom stopwords.")
except Exception as e:
    print(f"Warning: Error loading custom stopwords: {e}. Proceeding without custom stopwords.")
# --- End Load Custom Stopwords ---

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

    # Process Noun Chunks n-grams to capture phrases like "machine learning", "New York City"
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

# --- Helper Functions ---
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


def insert_into_database(postings_map, doc_freq_map, collection_freq_map, doc_metadata_list, processed_doc_ids_set, total_doc_count_final=0, is_final_batch=False):
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
        # --- Final Metadata Update ---
        if is_final_batch:
            print("  Updating final metadata (TotalDocuments)...")
            meta_sql = "INSERT INTO IndexMetadata (KeyName, Value) VALUES (%s, %s) ON DUPLICATE KEY UPDATE Value=VALUES(Value)"
            cursor.execute(meta_sql, ('TotalDocuments', total_doc_count_final))
            db.commit()
            print(f"  Updated TotalDocuments = {total_doc_count_final}.")
            return True # Done for final batch

        # --- Insert into Dictionary ---
        dict_data = []
        for term, df in doc_freq_map.items():
            cf = collection_freq_map.get(term, 0)
            if df > 0: # Should always be true if in doc_freq_map
                dict_data.append((term[:255], df, cf))

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
                print(f"  Inserted/Updated {len(dict_data)} terms in Dictionary.")
            except pymysql.Error as e:
                print(f"DB Error (Dictionary): {e}")
                db.rollback()
                return False

        # --- Insert into DocumentMetadata ---
        if doc_metadata_list:
            meta_sql = """
                INSERT INTO DocumentMetadata (DocID, RawWordCount)
                VALUES (%s, %s)
                ON DUPLICATE KEY UPDATE RawWordCount=VALUES(RawWordCount)
            """
            inserted_meta_count = 0
            try:
                print(f"  Preparing {len(doc_metadata_list)} DocumentMetadata entries...")
                for i in range(0, len(doc_metadata_list), DB_META_BATCH_SIZE):
                    batch = doc_metadata_list[i:i + DB_META_BATCH_SIZE]
                    cursor.executemany(meta_sql, batch)
                    inserted_meta_count += len(batch)
                    db.commit() # Commit metadata batches frequently
                    # print(f"    Inserted/Updated metadata batch {i//DB_META_BATCH_SIZE + 1}...") # Less verbose
                print(f"  Finished {inserted_meta_count} DocumentMetadata entries.")
            except pymysql.Error as e:
                 print(f"DB Error (DocumentMetadata): {e}")
                 db.rollback()
                 return False

        # --- Insert into Posting ---
        posting_data = []
        for term, postings_list in postings_map.items():
            truncated_term = term[:255]
            for doc_id, freq in postings_list:
                # Redundant check? processed_doc_ids_set should contain all relevant doc_ids
                # if doc_id in processed_doc_ids_set:
                truncated_doc_id = doc_id[:255]
                posting_data.append((truncated_term, truncated_doc_id, freq))

        if posting_data:
            posting_sql = """
                INSERT INTO Posting (Term, DocID, Term_Freq)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE Term_Freq = Term_Freq + VALUES(Term_Freq)
            """
            inserted_postings_count = 0
            try:
                print(f"  Preparing {len(posting_data)} Posting entries...")
                for i in range(0, len(posting_data), DB_POSTING_BATCH_SIZE):
                    batch = posting_data[i:i + DB_POSTING_BATCH_SIZE]
                    cursor.executemany(posting_sql, batch)
                    inserted_postings_count += len(batch)
                    # print(f"    Processed posting batch {i//DB_POSTING_BATCH_SIZE + 1}...") # Less verbose
                # Commit ONCE after all posting batches for the file
                db.commit()
                print(f"  Committed {inserted_postings_count} Posting entries.")
            except pymysql.Error as e:
                 print(f"DB Error (Posting): {e}")
                 db.rollback()
                 return False
        else:
             print("  No postings to insert for this batch.")

        # If all inserts succeeded for this batch
        return True

  except pymysql.Error as e:
        print(f"General DB Error in insert_into_database: {e}")
        try: db.rollback()
        except pymysql.Error as rb_err: print(f"Rollback failed: {rb_err}")
        return False
  except Exception as e:
        print(f"Unexpected Error in insert_into_database: {e}")
        return False

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

def db_worker(task_queue, results, worker_id):
    """Database worker thread processing database operations from a queue."""
    local_db = pymysql.connect(**DB_CONFIG)
    local_cursor = local_db.cursor()
    batch_count = 0
    success = True
    
    print(f"DB Worker {worker_id} started")
    
    try:
        while True:
            task = task_queue.get()
            if task is None:  # Sentinel to stop the thread
                break
                
            operation_type, sql, data = task
            
            try:
                local_cursor.executemany(sql, data)
                batch_count += 1
                
                # Commit periodically rather than for every operation
                if batch_count >= DB_COMMIT_INTERVAL:
                    local_db.commit()
                    batch_count = 0
                    print(f"  Worker {worker_id}: Committed {DB_COMMIT_INTERVAL} batches")
            except pymysql.Error as e:
                print(f"DB Worker {worker_id} Error: {e}")
                success = False
            finally:
                task_queue.task_done()
        
        # Final commit for any remaining operations
        if batch_count > 0:
            try:
                local_db.commit()
                print(f"  Worker {worker_id}: Final commit of {batch_count} batches")
            except pymysql.Error as e:
                print(f"DB Worker {worker_id} Error during final commit: {e}")
                success = False
    finally:
        local_cursor.close()
        local_db.close()
        print(f"DB Worker {worker_id} finished, connection closed")
        results[worker_id] = success

# --- Main execution ---
if __name__ == "__main__":
    main_start_time = time.time()
    json_directory = r'C:\Users\swoos\OneDrive\Documents\GitHub\CIS492_BigData_FinalProject\enwiki20201020'
    all_processed_doc_ids_global = set()
    total_files_processed_successfully = 0
    total_articles_processed = 0

    print(f"Scanning directory: {json_directory}")
    try:
        json_files = [f for f in os.listdir(json_directory) if f.endswith('.json')]
        json_files.sort()
        total_files_found = len(json_files)
        if total_files_found == 0: raise FileNotFoundError("No JSON files found.")
        print(f"Found {total_files_found} JSON files.")
    except Exception as e:
        print(f"Error scanning directory: {e}")
        exit(1)

    start_index = 54 # Start fresh run
    print(f"--- STARTING FRESH RUN from index {start_index} ---")

    for i, json_filename in enumerate(json_files[start_index:], start=start_index):
        file_start_time = time.time()
        # Reset in-memory structures for each file
        postings_batch = defaultdict(list)
        doc_freq_batch = defaultdict(int)
        collection_freq_batch = defaultdict(int)
        file_processed_doc_ids_batch = set()
        doc_metadata_batch = [] # Stores (doc_id, word_count)

        json_file_path = os.path.join(json_directory, json_filename)
        print(f"\n--- Processing file {i + 1}/{total_files_found}: {json_filename} ---")

        articles = load_json_data(json_file_path)
        load_end_time = time.time()

        if articles and isinstance(articles, list):
            print(f"Loaded {len(articles)} potential articles (Load time: {load_end_time - file_start_time:.2f}s)")

            # --- Prepare data for nlp.pipe (with length check) ---
            texts_to_process = []
            doc_ids_titles = []
            doc_metadata_pre_nlp = {} # Store metadata temporarily keyed by index
            initial_doc_metadata_batch = [] # Store metadata for skipped/empty docs separately

            print(f"Preprocessing articles (checking length < {MAX_TEXT_LENGTH})...")
            valid_article_count = 0
            skipped_length_count = 0
            skipped_other_count = 0
            for article_index, article in enumerate(articles):
                if not isinstance(article, dict):
                    skipped_other_count += 1
                    continue
                doc_id = article.get('id')
                text = article.get('text', '') # Default to empty string
                title = article.get('title', 'N/A')

                if not doc_id:
                    skipped_other_count += 1
                    continue

                # Calculate RawWordCount here
                word_count = len(re.split(r'\s+', text.strip())) if text.strip() else 0

                # Check text length BEFORE adding to NLP list
                if len(text) >= MAX_TEXT_LENGTH:
                    skipped_length_count += 1

                    initial_doc_metadata_batch.append((doc_id[:255], word_count))

                    # Still record metadata for skipped long docs
                    all_processed_doc_ids_global.add(doc_id) # Count as processed
                    file_processed_doc_ids_batch.add(doc_id)
                    continue

                # If valid and within length limit, add for NLP
                if text.strip():
                    current_idx = len(texts_to_process) # Index in the list being sent to nlp.pipe
                    texts_to_process.append(text)
                    doc_ids_titles.append({'id': doc_id, 'title': title})
                    doc_metadata_pre_nlp[current_idx] = (doc_id, word_count) # Store metadata by NLP index
                    valid_article_count += 1
                else:
                    # Record metadata for docs with ID but no text
                    initial_doc_metadata_batch.append((doc_id[:255], 0))
                    all_processed_doc_ids_global.add(doc_id) # Count as processed
                    file_processed_doc_ids_batch.add(doc_id)
                    skipped_other_count += 1

            print(f"Prepared {valid_article_count} articles for NLP. Skipped {skipped_length_count} (too long), {skipped_other_count} (other).")

            doc_metadata_batch.extend(initial_doc_metadata_batch)

            if not texts_to_process:
                print("No valid articles to process with NLP in this file.")
                # Insert any metadata collected from skipped docs
                if doc_metadata_batch:
                     print("Inserting metadata for skipped/empty articles...")
                     # Need to call insert_into_database just for metadata
                     insert_success = insert_into_database(
                         {}, {}, {}, doc_metadata_batch, file_processed_doc_ids_batch,
                         0,  
                         is_final_batch=False # Not final batch)
                     )
                     if not insert_success: print("Error inserting metadata for skipped docs.")

            # --- Process texts iteratively using nlp.pipe ---
            nlp_start_time = time.time()
            print(f"Starting iterative nlp.pipe() with batch_size={TARGET_NLP_BATCH_SIZE}...")
            processed_nlp_count = 0

            current_batch_size = TARGET_NLP_BATCH_SIZE
            try:
                print(f"Starting iterative nlp.pipe() with target batch_size={current_batch_size}...")
                # Iterate directly over the generator
                for idx, doc in enumerate(nlp.pipe(texts_to_process, batch_size=TARGET_NLP_BATCH_SIZE)):
                    # Get corresponding doc_id and metadata using the index 'idx'
                    doc_id, word_count = doc_metadata_pre_nlp[idx]
                    title = doc_ids_titles[idx]['title']

                    # Add doc_id to sets (might be redundant if already added, but safe)
                    file_processed_doc_ids_batch.add(doc_id)
                    all_processed_doc_ids_global.add(doc_id)
                    # Add metadata for this successfully processed doc
                    doc_metadata_batch.append((doc_id[:255], word_count))

                    # --- Extract tokens ---
                    final_tokens_for_index = []
                    for token in doc:
                        lemma = token.lemma_.lower()
                        if (not token.is_stop and
                            lemma not in custom_stopwords and
                            token.is_alpha and
                            len(lemma) > 1):
                            final_tokens_for_index.append(lemma)

                    for chunk in doc.noun_chunks:
                        lemmatized_chunk_parts = [token.lemma_.lower() for token in chunk if not token.is_stop and token.is_alpha and token.lemma_.lower() not in custom_stopwords]
                        if len(lemmatized_chunk_parts) > 1:
                            phrase = "_".join(lemmatized_chunk_parts)
                            if phrase and len(phrase) > 1:
                                final_tokens_for_index.append(phrase)
                    # --- End token extraction ---

                    if not final_tokens_for_index:
                        processed_nlp_count += 1
                        continue # No indexable tokens, but metadata was added

                    # --- Calculate term frequencies (TF) ---
                    # Use list before set() conversion for accurate TF counts
                    term_counts = Counter(final_tokens_for_index)

                    # --- Update batch index structures ---
                    processed_terms_in_doc = set()
                    for term, freq in term_counts.items():
                        truncated_term = term[:255]
                        truncated_doc_id = doc_id[:255]

                        postings_batch[truncated_term].append((truncated_doc_id, freq))
                        collection_freq_batch[truncated_term] += freq
                        if truncated_term not in processed_terms_in_doc:
                            doc_freq_batch[truncated_term] += 1
                            processed_terms_in_doc.add(truncated_term)
                    # --- End update batch ---

                    processed_nlp_count += 1
                    total_articles_processed += 1 # Count articles successfully processed by NLP

                    # Progress indicator
                    if processed_nlp_count % 500 == 0:
                        print(f"    Processed {processed_nlp_count}/{valid_article_count} articles via NLP...")

            except Exception as nlp_err:
                error_str = str(nlp_err).lower()
                is_memory_error = "memory" in error_str or "cuda" in error_str

                if is_memory_error and current_batch_size == TARGET_NLP_BATCH_SIZE:
                    print(f"\n!!! WARNING: Potential Memory Error with batch_size={TARGET_NLP_BATCH_SIZE}: {nlp_err} !!!")
                    # Use FALLBACK size
                    print(f"--- Attempting fallback with batch_size={FALLBACK_NLP_BATCH_SIZE} ---")
                    current_batch_size = FALLBACK_NLP_BATCH_SIZE # Set to FALLBACK

                    # --- Reset data collected during failed attempt ---
                    postings_batch = defaultdict(list)
                    doc_freq_batch = defaultdict(int)
                    collection_freq_batch = defaultdict(int)
                    doc_metadata_batch = list(initial_doc_metadata_batch) # Reset to only initial
                    processed_nlp_count = 0
                    # --- End Reset ---

                    # --- Retry with fallback batch size ---
                    try:
                        nlp_start_time = time.time() # Restart timer
                        print(f"Starting iterative nlp.pipe() with fallback batch_size={current_batch_size}...")
                        # --- Inner loop (Retry) ---
                        for idx, doc in enumerate(nlp.pipe(texts_to_process, batch_size=current_batch_size)):
                            # --- (Repeat processing logic as above) ---
                            doc_id, word_count = doc_metadata_pre_nlp[idx]
                            title = doc_ids_titles[idx]['title']
                            file_processed_doc_ids_batch.add(doc_id)
                            all_processed_doc_ids_global.add(doc_id)
                            doc_metadata_batch.append((doc_id[:255], word_count))
                            final_tokens_for_index = []
                            # ... (token extraction) ...
                            if not final_tokens_for_index:
                                processed_nlp_count += 1
                                continue
                            term_counts = Counter(final_tokens_for_index)
                            processed_terms_in_doc = set()
                            # ... (update batch index structures) ...
                            processed_nlp_count += 1
                            total_articles_processed += 1
                            if processed_nlp_count % 500 == 0: print(f"    Processed {processed_nlp_count}/{valid_article_count} articles via NLP (fallback)...")
                        # --- End Inner loop (Retry) ---
                        print(f"  Finished NLP processing loop successfully with fallback batch_size={current_batch_size} (Time: {time.time() - nlp_start_time:.2f}s)")

                    except Exception as fallback_err:
                        print(f"\n!!! ERROR: Fallback NLP processing failed with batch_size={current_batch_size}: {fallback_err} !!!")
                        print("--- Skipping NLP-derived data for this file ---")
                        postings_batch = defaultdict(list)
                        doc_freq_batch = defaultdict(int)
                        collection_freq_batch = defaultdict(int)
                        doc_metadata_batch = list(initial_doc_metadata_batch)

            # --- Database Insertion ---
            print(f"Inserting data for {len(file_processed_doc_ids_batch)} documents from {json_filename}...")
            db_start_time = time.time()
            insert_success = insert_into_database(
                postings_batch,
                doc_freq_batch,
                collection_freq_batch,
                doc_metadata_batch, # Pass combined metadata
                file_processed_doc_ids_batch,
                is_final_batch=False # total_doc_count_final is 0 here
            )
            db_end_time = time.time()
            print(f"  Finished database insertion attempt (Time: {db_end_time - db_start_time:.2f}s)")

            if insert_success:
                total_files_processed_successfully += 1
                print(f"Successfully processed and inserted data for {json_filename}")
            else:
                print(f"!!! Database insertion failed for {json_filename}. Check errors. !!!")

        # ... (Handle file load errors) ...

        file_end_time = time.time()
        print(f"Time taken for file {i + 1}: {file_end_time - file_start_time:.2f} seconds")
        print("-" * 60)

    # --- Final Metadata Update ---
    print("\n--- Performing final metadata update ---")
    final_doc_count = len(all_processed_doc_ids_global)
    final_update_success = insert_into_database({}, {}, {}, [], set(), total_doc_count_final=final_doc_count, is_final_batch=True)
    if not final_update_success: print("!!! Error during final metadata update. !!!")

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
