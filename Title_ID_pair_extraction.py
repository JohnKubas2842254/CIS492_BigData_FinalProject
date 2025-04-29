import json
import os
import pymysql #type: ignore
import time

# --- Database Setup (Connect to the ORIGINAL database) ---
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "2842254K", # Replace with your password
    "database": "BigData_FinalProject_spaCy", # <--- ORIGINAL DB
    "charset": "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor # Use DictCursor if needed, otherwise default is fine
}

# --- Configuration ---
JSON_DIRECTORY = r'C:\Users\swoos\OneDrive\Documents\GitHub\CIS492_BigData_FinalProject\enwiki20201020'
BATCH_SIZE = 10000 # How many records to insert into DB at once
TABLE_NAME = "DocumentTitles"
# --- End Configuration ---

# --- Helper Function (Copied/Adapted from Indexer) ---
def load_json_data(filepath):
    """Loads JSON data from a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            try:
                # Handle potential single large JSON object/array
                data = json.loads(content)
                # Ensure it's a list, even if it was a single object
                if not isinstance(data, list):
                    data = [data] # Wrap single object in a list
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

# --- Main Execution ---
if __name__ == "__main__":
    main_start_time = time.time()

    print(f"Scanning directory: {JSON_DIRECTORY}")
    try:
        json_files = [f for f in os.listdir(JSON_DIRECTORY) if f.endswith('.json')]
        json_files.sort() # Ensure consistent order
        total_files_found = len(json_files)
        if total_files_found == 0:
            print("No JSON files found in the directory. Exiting.")
            exit()
        print(f"Found {total_files_found} JSON files to process.")
    except FileNotFoundError:
        print(f"Error: Directory not found: {JSON_DIRECTORY}")
        exit(1)
    except Exception as e:
        print(f"Error scanning directory: {e}")
        exit(1)

    db = None
    cursor = None
    total_articles_processed = 0
    total_titles_inserted = 0
    batch_data = []

    try:
        print(f"Connecting to database '{DB_CONFIG['database']}'...")
        db = pymysql.connect(**DB_CONFIG)
        cursor = db.cursor()
        cursor.execute(f"USE {DB_CONFIG['database']}")
        print("Database connection established.")

        # Ensure DocumentTitles table exists
        print(f"Ensuring '{TABLE_NAME}' table exists...")
        # Using VARCHAR(512) for Title for more space, adjust if needed
        # Using utf8mb4_unicode_ci for better sorting/comparison if needed
        cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            DocID VARCHAR(255) PRIMARY KEY,
            Title VARCHAR(512) NULL
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """)
        print(f"Table '{TABLE_NAME}' is ready.")

        # SQL for insertion - INSERT IGNORE skips duplicates based on PRIMARY KEY (DocID)
        insert_sql = f"INSERT IGNORE INTO {TABLE_NAME} (DocID, Title) VALUES (%s, %s)"

        # Process files
        for i, json_filename in enumerate(json_files):
            file_start_time = time.time()
            json_file_path = os.path.join(JSON_DIRECTORY, json_filename)
            print(f"\n--- Processing file {i + 1}/{total_files_found}: {json_filename} ---")

            articles = load_json_data(json_file_path)

            if articles and isinstance(articles, list):
                file_article_count = 0
                for article in articles:
                    if not isinstance(article, dict):
                        continue
                    doc_id = article.get('id')
                    title = article.get('title') # Get title, could be None

                    if doc_id: # Only need DocID to exist
                        # Truncate if necessary to fit table definition
                        truncated_doc_id = doc_id[:255]
                        truncated_title = title[:512] if title else None # Handle None title

                        batch_data.append((truncated_doc_id, truncated_title))
                        file_article_count += 1

                        # Insert batch if full
                        if len(batch_data) >= BATCH_SIZE:
                            try:
                                inserted_count = cursor.executemany(insert_sql, batch_data)
                                db.commit()
                                total_titles_inserted += inserted_count
                                print(f"  Inserted batch of {len(batch_data)} (New: {inserted_count}). Total inserted: {total_titles_inserted}")
                                batch_data = [] # Clear batch
                            except pymysql.Error as e:
                                print(f"Database Error during batch insert: {e}")
                                db.rollback()
                                # Optional: Decide whether to continue or exit on error
                                # exit(1)
                    else:
                         # print(f"  Skipping article due to missing ID.") # Optional logging
                         pass

                total_articles_processed += file_article_count
                file_end_time = time.time()
                print(f"  Finished file {i + 1}. Processed {file_article_count} articles. Time: {file_end_time - file_start_time:.2f}s")

            else:
                print(f"  Skipping file {json_filename} due to load error or empty content.")

        # Insert any remaining data
        if batch_data:
            try:
                print("\nInserting final batch...")
                inserted_count = cursor.executemany(insert_sql, batch_data)
                db.commit()
                total_titles_inserted += inserted_count
                print(f"  Inserted final batch of {len(batch_data)} (New: {inserted_count}). Total inserted: {total_titles_inserted}")
            except pymysql.Error as e:
                print(f"Database Error during final batch insert: {e}")
                db.rollback()

    except pymysql.Error as e:
        print(f"\nDatabase connection or execution error: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    finally:
        if cursor:
            cursor.close()
        if db:
            db.close()
            print("\nDatabase connection closed.")

    main_end_time = time.time()
    print(f"\n--- Extraction Complete ---")
    print(f"Total articles processed: {total_articles_processed}")
    print(f"Total unique titles inserted/found: {total_titles_inserted}")
    print(f"Total time: {main_end_time - main_start_time:.2f} seconds")
