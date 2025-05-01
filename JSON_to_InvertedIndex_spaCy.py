import json
import spacy #type: ignore
import re
import os
from collections import defaultdict, Counter
import pymysql #type: ignore
import time
import csv
import gc
import threading
from queue import Queue
import multiprocessing
import tempfile
import torch #type: ignore

# --- Configuration ---
# Set this to the number of CPU cores you want to use for NLP processing
# Usually (CPU cores - 1) is a good setting to leave resources for the OS
NLP_THREAD_COUNT = max(1, multiprocessing.cpu_count() - 1)
MAX_TEXT_LENGTH = 1000000 
TARGET_NLP_BATCH_SIZE = 256 
FALLBACK_NLP_BATCH_SIZE = 64  
DB_POSTING_BATCH_SIZE = 500000 
DB_META_BATCH_SIZE = 10000 
DB_COMMIT_INTERVAL = 100 
# --- End Configuration ---

# --- Database optimization queries ---
DB_OPTIMIZATION_QUERIES = [
    # Maximum performance settings - USE WITH CAUTION
    "SET GLOBAL innodb_flush_log_at_trx_commit = 0",  # Maximum speed (0 instead of 2)
    "SET GLOBAL sync_binlog = 0",  # Disable binary log syncing
    
    # Buffer configurations
    "SET GLOBAL innodb_buffer_pool_size = 4294967296",  # 4GB buffer (adjust based on RAM)
    "SET GLOBAL innodb_log_buffer_size = 33554432",  # 32MB
    "SET GLOBAL max_allowed_packet = 268435456",  # 256MB
    
    # Session settings
    "SET SESSION bulk_insert_buffer_size = 1073741824",  # 1GB
    "SET SESSION sort_buffer_size = 67108864",  # 64MB
    
    # Performance settings
    "SET GLOBAL innodb_doublewrite = 0",  # Disable doublewrite buffer (RISKY but FAST)
    "SET GLOBAL innodb_flush_method = 'O_DIRECT'",  # Bypass file system cache
    
    # Transaction settings
    "SET autocommit=0",
    "SET unique_checks=0",
    "SET foreign_key_checks=0"
]

# Load NLP model once during main process initialization
try:
    spacy.require_gpu()
    print("GPU available, using GPU.")
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
except Exception as e:
    print(f"GPU not available or CuPy/spaCy GPU support not installed ({e}). Using CPU.")
    nlp = spacy.load("en_core_web_sm", disable=["ner"]) # Disable NER to save memory

def handle_gpu_error(e):
    """Handle GPU out of memory errors by falling back to CPU"""
    error_msg = str(e).lower()
    is_gpu_error = any(x in error_msg for x in ["cuda", "gpu", "memory", "out of memory", "oom"])
    
    if is_gpu_error:
        print("GPU memory error detected. Attempting to switch to CPU...")
        try:
            # Try to explicitly free CUDA memory
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("Cleared CUDA cache")
        except:
            pass
        
        try:
            # Disable spaCy GPU usage globally
            spacy.prefer_cpu()
            print("Switched to CPU mode for spaCy")
            return True
        except:
            print("Failed to switch to CPU mode")
    
    return False

# --- Optimize NLP model ---
def optimize_nlp_model():
    # Keep only the components we need for tokenization and lemmatization
    components_to_keep = ['tok2vec', 'tagger', 'attribute_ruler', 'lemmatizer', 'parser']
    if "parser" not in nlp.pipe_names:
        components_to_keep.append('parser')
    
    # Disable all other components
    for component in nlp.pipe_names:
        if component not in components_to_keep:
            nlp.disable_pipe(component)
    
    print(f"Optimized NLP model. Active components: {', '.join(nlp.pipe_names)}")

optimize_nlp_model()

def process_batch_with_nlp(texts, batch_size=TARGET_NLP_BATCH_SIZE):
    """Process a batch of texts with NLP in a thread-safe way"""
    results = []
    # Process in sub-batches to avoid memory issues
    for i in range(0, len(texts), batch_size):
        sub_batch = texts[i:i+batch_size]
        try:
            docs = list(nlp.pipe(sub_batch, batch_size=batch_size))
            results.extend(docs)
            
            # Force garbage collection after each sub-batch
            if i % 1000 == 0 and i > 0:
                gc.collect()
                
        except Exception as e:
            print(f"Error in NLP batch processing: {e}")
            # Check if it's a GPU error first
            if handle_gpu_error(e):
                # If it was a GPU error and we switched to CPU, retry
                try:
                    print("Retrying with CPU after GPU error...")
                    docs = list(nlp.pipe(sub_batch, batch_size=FALLBACK_NLP_BATCH_SIZE))
                    results.extend(docs)
                    continue
                except Exception as cpu_e:
                    print(f"CPU fallback also failed: {cpu_e}")
            
            # Otherwise continue with batch size reduction
            if batch_size > FALLBACK_NLP_BATCH_SIZE:
                print(f"Retrying with smaller batch size {FALLBACK_NLP_BATCH_SIZE}")
                sub_results = process_batch_with_nlp(sub_batch, FALLBACK_NLP_BATCH_SIZE)
                results.extend(sub_results)
            else:
                # If already at minimum batch size, process one by one
                for text in sub_batch:
                    try:
                        doc = nlp(text)
                        results.append(doc)
                    except:
                        # If individual processing fails, add None as placeholder
                        results.append(None)
    return results

def optimize_database(connection):
    """Apply database optimizations for bulk inserts"""
    try:
        cursor = connection.cursor()
        for query in DB_OPTIMIZATION_QUERIES:
            try:
                cursor.execute(query)
            except pymysql.Error as e:
                # Some optimizations might require admin privileges, so just warn
                print(f"Warning: Could not apply optimization: {query} - {e}")
        connection.commit()
        print("Database optimizations applied where possible")
    except Exception as e:
        print(f"Error applying database optimizations: {e}")

# --- Database Setup Function ---
def setup_database():
    try:
        db = pymysql.connect(**DB_CONFIG)
        cursor = db.cursor()
        
        # Initialize the database
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']}")
        cursor.execute(f"USE {DB_CONFIG['database']}")
        
        # Create tables (your original table creation code)
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
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin 
        ROW_FORMAT=COMPRESSED 
        KEY_BLOCK_SIZE=8;
        """)

        # Execute this after table creation
        cursor.execute("""
        ALTER TABLE Posting 
        ADD INDEX idx_docid (DocID),
        STATS_PERSISTENT=1, 
        STATS_AUTO_RECALC=1;
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
        
        optimize_database(db)
        print(f"Database '{DB_CONFIG['database']}' initialized successfully.")
        
        return db, cursor
    except pymysql.Error as e:
        print(f"Database connection or setup failed: {e}")
        exit(1)

def save_to_csv(data, prefix, headers):
    """Save data to a temporary CSV file"""
    fd, path = tempfile.mkstemp(suffix='.csv', prefix=prefix)
    with os.fdopen(fd, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(data)
    return path

def load_csv_to_database(db, cursor, csv_path, table, columns, batch_size=10000):
    """Load data from CSV into database"""
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        batch = []
        
        placeholders = ', '.join(['%s'] * len(columns))
        sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
        
        if table == "Posting":
            sql += " ON DUPLICATE KEY UPDATE Term_Freq = Term_Freq + VALUES(Term_Freq)"
        elif table == "Dictionary":
            sql += " ON DUPLICATE KEY UPDATE TotalDocsFreq = TotalDocsFreq + VALUES(TotalDocsFreq), TotalCollectionFreq = TotalCollectionFreq + VALUES(TotalCollectionFreq)"
        
        for i, row in enumerate(reader):
            batch.append(row)
            if len(batch) >= batch_size:
                cursor.executemany(sql, batch)
                if i % (batch_size*10) == 0:
                    db.commit()
                batch = []
                
        if batch:
            cursor.executemany(sql, batch)
            db.commit()
    
    # Remove temp file
    try:
        os.remove(csv_path)
    except:
        pass

# Add this function after load_csv_to_database
def calculate_magnitude_sq(db, cursor, doc_id, term_freqs):
    """Calculate and store the squared magnitude for a document vector"""
    # Calculate the sum of squared term frequencies (TF)
    mag_sq = sum(freq * freq for freq in term_freqs.values())
    
    # Update the DocumentMetadata table
    update_sql = """
    UPDATE DocumentMetadata 
    SET MagnitudeSq = %s 
    WHERE DocID = %s
    """
    cursor.execute(update_sql, (mag_sq, doc_id))

def process_file_pipeline(file_index, file_path, db, cursor):
    """Process a file in pipeline stages"""
    # Stage 1: Load and NLP process
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            articles = json.loads(f.read())
        
        (postings_batch, doc_freq_batch, collection_freq_batch, 
         doc_metadata_batch, file_processed_doc_ids) = process_file_with_nlp(articles)
        
        # Stage 2: Save to CSVs
        dict_data = [(term[:255], df, collection_freq_batch.get(term, 0)) 
                    for term, df in doc_freq_batch.items()]
        
        posting_data = []
        for term, postings in postings_batch.items():
            term = term[:255]
            for doc_id, freq in postings:
                posting_data.append((term, doc_id[:255], freq))
        
        # Calculate magnitude squared for each document
        doc_term_freqs = {}
        for term, postings in postings_batch.items():
            for doc_id, freq in postings:
                if doc_id not in doc_term_freqs:
                    doc_term_freqs[doc_id] = {}
                doc_term_freqs[doc_id][term] = freq

        # Add MagnitudeSq to metadata_csv
        doc_metadata_with_mag = []
        for doc_id, word_count in doc_metadata_batch:
            # Calculate magnitude squared (sum of squared term frequencies)
            mag_sq = 0
            if doc_id in doc_term_freqs:
                mag_sq = sum(freq * freq for freq in doc_term_freqs[doc_id].values())
            doc_metadata_with_mag.append((doc_id, word_count, mag_sq))

        total_postings = sum(len(postings) for postings in postings_batch.values())
        print(f"Generated {len(dict_data)} unique terms and {total_postings} posting entries")
        # Export to CSVs
        dict_csv = save_to_csv(dict_data, f"dict_{file_index}_", ["Term", "TotalDocsFreq", "TotalCollectionFreq"])
        posting_csv = save_to_csv(posting_data, f"posting_{file_index}_", ["Term", "DocID", "Term_Freq"])
        metadata_csv = save_to_csv(doc_metadata_with_mag, f"meta_{file_index}_", ["DocID", "RawWordCount", "MagnitudeSq"])        
        # Stage 3: Database upload (this can run in background)
        def upload_background():
            try:
                print(f"Starting background upload for file {file_index + 1}")
                
                # Track timing for diagnostics
                upload_start_time = time.time()
                
                try:
                    local_db = pymysql.connect(**DB_CONFIG)
                    print(f"  DB connection established ({time.time() - upload_start_time:.2f}s)")
                except Exception as conn_err:
                    print(f"  ERROR connecting to database: {conn_err}")
                    return
                    
                try:
                    local_cursor = local_db.cursor()
                    
                    # Apply optimizations to this connection
                    for query in DB_OPTIMIZATION_QUERIES:
                        try:
                            local_cursor.execute(query)
                        except Exception as opt_err:
                            pass  # Already reporting these elsewhere
                    print(f"  DB optimizations applied ({time.time() - upload_start_time:.2f}s)")
                except Exception as cursor_err:
                    print(f"  ERROR setting up cursor/optimizations: {cursor_err}")
                    local_db.close()
                    return
                    
                # Perform each database operation with detailed error reporting
                try:
                    print(f"  Loading Dictionary data from {os.path.basename(dict_csv)}")
                    dict_start = time.time()
                    load_csv_to_database(local_db, local_cursor, dict_csv, "Dictionary", 
                                    ["Term", "TotalDocsFreq", "TotalCollectionFreq"])
                    print(f"  Dictionary loaded ({time.time() - dict_start:.2f}s)")
                except Exception as dict_err:
                    print(f"  ERROR loading Dictionary data: {dict_err}")
                    try:
                        local_db.rollback()
                    except:
                        pass
                
                try:
                    print(f"  Loading DocumentMetadata data from {os.path.basename(metadata_csv)}")
                    meta_start = time.time()
                    load_csv_to_database(local_db, local_cursor, metadata_csv, "DocumentMetadata",
                                        ["DocID", "RawWordCount", "MagnitudeSq"])
                    print(f"  DocumentMetadata loaded ({time.time() - meta_start:.2f}s)")
                except Exception as meta_err:
                    print(f"  ERROR loading DocumentMetadata: {meta_err}")
                    try:
                        local_db.rollback()
                    except:
                        pass
                
                try:
                    print(f"  Loading Posting data from {os.path.basename(posting_csv)}")
                    post_start = time.time()
                    load_csv_to_database(local_db, local_cursor, posting_csv, "Posting",
                                        ["Term", "DocID", "Term_Freq"], batch_size=DB_POSTING_BATCH_SIZE)
                    print(f"  Posting data loaded ({time.time() - post_start:.2f}s)")
                except Exception as post_err:
                    print(f"  ERROR loading Posting data: {post_err}")
                    try:
                        local_db.rollback()
                    except:
                        pass
                
                try:
                    # Final commit
                    commit_start = time.time()
                    local_db.commit()
                    print(f"  Final commit successful ({time.time() - commit_start:.2f}s)")
                except Exception as commit_err:
                    print(f"  ERROR during final commit: {commit_err}")
                
                try:
                    local_cursor.close()
                    local_db.close()
                except:
                    pass
                    
                print(f"Background upload completed for file {file_index + 1} in {time.time() - upload_start_time:.2f}s")
            except Exception as e:
                print(f"ERROR in background upload (uncaught): {e}")
                import traceback
                traceback.print_exc()  # Print the full stack trace
        
        # Start background upload
        upload_thread = threading.Thread(target=upload_background)
        upload_thread.daemon = True
        upload_thread.start()
        
        return file_processed_doc_ids, len(file_processed_doc_ids), upload_thread
    
    except Exception as e:
        print(f"Error in pipeline processing: {e}")
        return set(), 0, None

# --- Multithreaded NLP Processing ---
def process_file_with_nlp(articles, max_workers=NLP_THREAD_COUNT):
    """Process a file with multithreaded NLP"""
    import concurrent.futures
    
    # Prepare data
    valid_articles = []
    doc_ids = []
    word_counts = []
    titles = []
    skipped_length_count = 0
    skipped_other_count = 0
    skipped_docs_metadata = []
    
    # Filter valid articles first
    for article in articles:
        if not isinstance(article, dict):
            skipped_other_count += 1
            continue
        
        doc_id = article.get('id')
        text = article.get('text', '')
        title = article.get('title', 'N/A')
        
        if not doc_id or not text.strip():
            skipped_other_count += 1
            if doc_id:
                skipped_docs_metadata.append((doc_id[:255], 0))
            continue
            
        # Check length
        if len(text) >= MAX_TEXT_LENGTH:
            skipped_length_count += 1
            word_count = len(re.split(r'\s+', text.strip()))
            skipped_docs_metadata.append((doc_id[:255], word_count))
            continue
            
        # Calculate word count
        word_count = len(re.split(r'\s+', text.strip()))
        
        # Add to processing lists
        valid_articles.append(text)
        doc_ids.append(doc_id)
        word_counts.append(word_count)
        titles.append(title)
    
    print(f"Prepared {len(valid_articles)} articles for NLP. Skipped {skipped_length_count} (too long), {skipped_other_count} (other).")
    
    if not valid_articles:
        return {}, {}, {}, skipped_docs_metadata, set()
    
    # Process in batches with threads
    postings_batch = defaultdict(list)
    doc_freq_batch = defaultdict(int)
    collection_freq_batch = defaultdict(int)
    processed_doc_ids = set()
    doc_metadata_batch = list(skipped_docs_metadata)  # Start with skipped docs
    
    batch_size = min(5000, len(valid_articles))  # Process in manageable chunks
    
    for batch_start in range(0, len(valid_articles), batch_size):
        batch_end = min(batch_start + batch_size, len(valid_articles))
        print(f"Processing NLP batch {batch_start//batch_size + 1}/{(len(valid_articles)+batch_size-1)//batch_size}...")
        
        # Get the batch texts and metadata
        batch_texts = valid_articles[batch_start:batch_end]
        batch_ids = doc_ids[batch_start:batch_end]
        batch_counts = word_counts[batch_start:batch_end]
        
        # Process the batch with NLP
        docs = process_batch_with_nlp(batch_texts)
        
        # Process the results
        for i, doc in enumerate(docs):
            if doc is None:  # Skip failed processing
                continue
                
            doc_id = batch_ids[i]
            word_count = batch_counts[i]
            
            # Extract tokens
            final_tokens = []
            for token in doc:
                lemma = token.lemma_.lower()
                if (not token.is_stop and token.is_alpha and 
                    len(lemma) > 1 and lemma not in custom_stopwords):
                    final_tokens.append(lemma)
            
            # Process noun chunks if we have a parser
            if 'parser' in nlp.pipe_names:
                for chunk in doc.noun_chunks:
                    lemmatized_parts = [t.lemma_.lower() for t in chunk 
                                       if not t.is_stop and t.is_alpha and 
                                          t.lemma_.lower() not in custom_stopwords]
                    if len(lemmatized_parts) > 1:
                        phrase = "_".join(lemmatized_parts)
                        if phrase and len(phrase) > 1:
                            final_tokens.append(phrase)
            
            # Skip if no tokens extracted
            if not final_tokens:
                doc_metadata_batch.append((doc_id[:255], word_count))
                processed_doc_ids.add(doc_id)
                continue
                
            # Calculate term frequencies
            term_counts = Counter(final_tokens)
            
            # Update index structures
            processed_terms = set()
            for term, freq in term_counts.items():
                term = term[:255]  # Truncate if needed
                
                postings_batch[term].append((doc_id[:255], freq))
                collection_freq_batch[term] += freq
                
                if term not in processed_terms:
                    doc_freq_batch[term] += 1
                    processed_terms.add(term)
            
            # Add metadata and mark as processed
            doc_metadata_batch.append((doc_id[:255], word_count))
            processed_doc_ids.add(doc_id)
            
        # Free memory after each batch
        gc.collect()
            
    print(f"Successfully processed {len(processed_doc_ids)} documents with NLP")
    return postings_batch, doc_freq_batch, collection_freq_batch, doc_metadata_batch, processed_doc_ids

if __name__ == "__main__":
    main_start_time = time.time()
    json_directory = r'C:\Users\swoos\OneDrive\Documents\GitHub\CIS492_BigData_FinalProject\enwiki20201020'
    all_processed_doc_ids_global = set()
    total_files_processed_successfully = 0
    total_articles_processed = 0
    
    # Setup database
    DB_CONFIG = {
        "host": "localhost",
        "user": "root",
        "password": "2842254K", 
        "database": "BigData_FinalProject_spaCy2",
        "charset": "utf8mb4",
        "cursorclass": pymysql.cursors.DictCursor,
        "connect_timeout": 300, 
        "read_timeout": 600,
        "write_timeout": 600
    }
    
    db, cursor = setup_database()
    
    # Load custom stopwords
    CUSTOM_STOPWORDS_FILE = r'C:\Users\swoos\OneDrive\Documents\GitHub\CIS492_BigData_FinalProject\common_terms2.csv'
    custom_stopwords = set()
    try:
        with open(CUSTOM_STOPWORDS_FILE, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if row:
                    custom_stopwords.add(row[0])
        print(f"Loaded {len(custom_stopwords)} custom stopwords")
    except Exception as e:
        print(f"Warning: Error loading stopwords: {e}")
    
    # Process files
    json_files = sorted([f for f in os.listdir(json_directory) if f.endswith('.json')])
    start_index = 0 
    
    print(f"Found {len(json_files)} JSON files. Starting from index {start_index}.")
    
    active_threads = []
    max_active_threads = 3

    # Main processing loop
    for i, json_filename in enumerate(json_files[start_index:], start=start_index):
    # Wait if too many active threads
        while len(active_threads) >= max_active_threads:
            for t in list(active_threads):
                if not t.is_alive():
                    active_threads.remove(t)
            if len(active_threads) >= max_active_threads:
                time.sleep(1)
        
        file_start_time = time.time()
        json_file_path = os.path.join(json_directory, json_filename)
        print(f"\n--- Processing file {i+1}/{len(json_files)}: {json_filename} ---")
        
        # Process file in pipeline
        processed_doc_ids, doc_count, upload_thread = process_file_pipeline(
            i, json_file_path, db, cursor
        )
        
        # Track statistics
        if doc_count > 0:
            all_processed_doc_ids_global.update(processed_doc_ids)
            total_articles_processed += doc_count
            total_files_processed_successfully += 1
        
        if upload_thread:
            active_threads.append(upload_thread)
        
        file_end_time = time.time()
        print(f"Time taken for file {i+1}: {file_end_time - file_start_time:.2f} seconds")
        print("-" * 60)

    # Wait for all remaining uploads to complete
    print("Waiting for remaining uploads to complete...")
    for t in active_threads:
        t.join()
    
    # Summary
    print("\n--- Total Processing Summary ---")
    print(f"Successfully processed {total_files_processed_successfully}/{len(json_files[start_index:])} JSON files")
    print(f"Total articles processed: {total_articles_processed}")
    
    # Close database
    if db.open:
        cursor.close()
        db.close()
        print("Database connection closed")
    
    main_end_time = time.time()
    print(f"\nTotal execution time: {main_end_time - main_start_time:.2f} seconds")