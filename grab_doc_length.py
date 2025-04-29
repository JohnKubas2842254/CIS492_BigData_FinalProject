import json
import os
import pymysql #type: ignore
import time
import re # Import re for splitting

# --- Database Setup ---
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "2842254K", # Replace with your password
    "database": "BigData_FinalProject_spaCy", # Use the spaCy DB
    "charset": "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor
}

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
                    data = [data]
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

    # --- Configuration ---
    json_directory = r'C:\Users\swoos\OneDrive\Documents\GitHub\CIS492_BigData_FinalProject\enwiki20201020'
    batch_size = 5000 # How many records to insert into DB at once
    # --- End Configuration ---

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

    db = None
    cursor = None
    total_articles_processed = 0
    batch_data = []

    try:
        db = pymysql.connect(**DB_CONFIG)
        cursor = db.cursor()
        cursor.execute(f"USE {DB_CONFIG['database']}")
        print("Database connection established.")

        # Ensure DocumentMetadata table exists
        print("Ensuring DocumentMetadata table exists...")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS DocumentMetadata (
            DocID VARCHAR(255) PRIMARY KEY,
            RawWordCount INT,
            MagnitudeSq DOUBLE NULL
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)
        # Optional: Clear table if you want a fresh start each time
        # print("Clearing existing data from DocumentMetadata...")
        # cursor.execute("DELETE FROM DocumentMetadata")
        db.commit()
        print("DocumentMetadata table ready.")

        # --- Process Files ---
        for i, json_filename in enumerate(json_files):
            file_start_time = time.time()
            json_file_path = os.path.join(json_directory, json_filename)
            print(f"\n--- Processing file {i + 1}/{total_files_found}: {json_filename} ---")

            articles = load_json_data(json_file_path)

            if articles and isinstance(articles, list):
                print(f"  Loaded {len(articles)} potential articles.")
                articles_in_file = 0
                for article in articles:
                    if not isinstance(article, dict):
                        # print(f"  Skipping item: not a dictionary.")
                        continue # Skip non-dict items silently

                    doc_id = article.get('id')
                    text = article.get('text', '') # Default to empty string if 'text' key is missing

                    if not doc_id:
                        # print(f"  Skipping article (Title: '{article.get('title', 'N/A')}') due to missing ID.")
                        continue # Skip if no ID

                    # Calculate simple word count using regex split for better handling of punctuation
                    # This splits on whitespace characters
                    word_count = len(re.split(r'\s+', text.strip())) if text.strip() else 0

                    # Add to batch for insertion (Truncate DocID if necessary)
                    batch_data.append((doc_id[:255], word_count))
                    total_articles_processed += 1
                    articles_in_file += 1

                    # Insert batch into database when full
                    if len(batch_data) >= batch_size:
                        insert_sql = """
                            INSERT INTO DocumentMetadata (DocID, RawWordCount)
                            VALUES (%s, %s)
                            ON DUPLICATE KEY UPDATE RawWordCount=VALUES(RawWordCount)
                        """
                        try:
                            rows_affected = cursor.executemany(insert_sql, batch_data)
                            db.commit()
                            print(f"    Inserted/Updated batch of {len(batch_data)} lengths (affected {rows_affected} rows). Total articles processed: {total_articles_processed}")
                            batch_data = [] # Clear batch
                        except pymysql.Error as e:
                            print(f"Database Error during batch insert: {e}")
                            db.rollback()
                            # Consider adding logic to retry or stop here

                file_end_time = time.time()
                print(f"  Finished processing {articles_in_file} articles from file in {file_end_time - file_start_time:.2f}s")
            else:
                print(f"  Skipping file {json_filename} due to load error or invalid format.")

        # --- Insert Final Batch ---
        if batch_data:
            print("\nInserting final batch...")
            insert_sql = """
                INSERT INTO DocumentMetadata (DocID, RawWordCount)
                VALUES (%s, %s)
                ON DUPLICATE KEY UPDATE RawWordCount=VALUES(RawWordCount)
            """
            try:
                rows_affected = cursor.executemany(insert_sql, batch_data)
                db.commit()
                print(f"  Inserted/Updated final batch of {len(batch_data)} lengths (affected {rows_affected} rows).")
            except pymysql.Error as e:
                print(f"Database Error during final batch insert: {e}")
                db.rollback()

        print("\n--- Length Calculation Summary ---")
        print(f"Total articles processed for length: {total_articles_processed}")

    except pymysql.Error as e:
        print(f"\nDatabase connection or operation failed: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    finally:
        # --- Close Database Connection ---
        if cursor:
            try:
                cursor.close()
            except Exception as e:
                 print(f"Error closing cursor: {e}")
        if db and db.open:
            try:
                db.close()
                print("Database connection closed.")
            except Exception as e:
                 print(f"Error closing database connection: {e}")

    main_end_time = time.time()
    print(f"\nTotal execution time: {main_end_time - main_start_time:.2f} seconds")