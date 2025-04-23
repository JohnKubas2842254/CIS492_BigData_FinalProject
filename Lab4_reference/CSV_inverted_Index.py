import csv
import re
import pymysql  # type: ignore
from collections import defaultdict
import sys
import spacy  # type: ignore
import math

"""Increase the CSV field size limit, there were a few rows with long text
This is a workaround for the CSV module's default limit on field size """
csv.field_size_limit(sys.maxsize)

# MySQL connection setup
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "2842254K",
    "database": "BigDataLab4",
    "charset": "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor
}

db = pymysql.connect(**DB_CONFIG)
cursor = db.cursor()

# Create MySQL tables
cursor.execute("""
CREATE TABLE IF NOT EXISTS Dictionary (
    Term VARCHAR(255) PRIMARY KEY,
    TotalDocsFreq INT,
    TotalCollectionFreq INT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS Posting (
    Term VARCHAR(255),
    DocID VARCHAR(255),
    Term_Freq INT,
    PRIMARY KEY (Term, DocID)
)
""")
#New for Part 2: Similarity Table
cursor.execute("""
CREATE TABLE IF NOT EXISTS Similarity (
    DocID1 VARCHAR(255),
    DocID2 VARCHAR(255),
    Similarity FLOAT,
    PRIMARY KEY (DocID1, DocID2)
)
""")

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
    print("spaCy NLP model loaded successfully!")
except (ImportError, OSError) as e:
    SPACY_AVAILABLE = False
    print(f"Warning: Could not load spaCy model: {e}")

# Function to tokenize text using re
def nlp_pipeline(text):
    """Process text through a complete NLP pipeline with enhanced token cleaning"""
    # Process with spaCy - applying the full pipeline in sequence
    doc = nlp(text)

    """Define a pattern for characters to keep (alphanumeric, apostrophe, hyphen)
    This will be used to clean tokens aggressively, throughout testing I ran into
    many cases where characters were not being removed correctly """
    keep_chars_pattern = re.compile(r"[^a-z0-9'/-]")

    # 1. Extract and clean tokens
    tokens = []
    for token in doc:
        # Basic spaCy filtering
        if not token.is_punct and not token.is_space:
            # Lowercase and strip leading/trailing whitespace FIRST
            original_token = token.text
            clean_token = token.text.lower().strip()
            # Remove unwanted characters using regex, keep apostrophes/hyphens
            clean_token = keep_chars_pattern.sub('', clean_token)
            # Final strip for any leftover punctuation/whitespace from regex cleaning
            clean_token = clean_token.strip('-\'')
            # Check if the token is valid after cleaning
            if clean_token and (len(clean_token) > 1 or clean_token.isalnum()):
                tokens.append(clean_token)

    # 2. Apply lemmatization
    lemmas = []
    for token in doc:
        # Basic spaCy filtering
        if not token.is_punct and not token.is_space:
            # Get lemma and lowercase
            lemma = token.lemma_.lower().strip()
            # Apply the same cleaning as tokens
            clean_lemma = keep_chars_pattern.sub('', lemma)
            # Final strip
            clean_lemma = clean_lemma.strip('-\'')
            # Check if the lemma is valid after cleaning
            # Ensure it's not just symbols or empty after cleaning
            if clean_lemma and (len(clean_lemma) > 1 or clean_lemma.isalnum()):
                 # Additional check: Ensure it contains at least one letter or number
                 if any(c.isalnum() for c in clean_lemma):
                    lemmas.append(clean_lemma)

    # 3. Extract POS tags (using cleaned tokens)
    pos_tags = []
    # Re-iterate or use cleaned tokens if needed, ensuring alignment
    for token in doc:
         if not token.is_punct and not token.is_space:
            clean_text = token.text.lower().strip()
            clean_text = keep_chars_pattern.sub('', clean_text)
            clean_text = clean_text.strip('-\'')
            if clean_text and (len(clean_text) > 1 or clean_text.isalnum()):
                 if any(c.isalnum() for c in clean_text):
                    pos_tags.append((clean_text, token.pos_))

    # 4. NER, Extract named entities
    entities = [(ent.text.strip(), ent.label_) for ent in doc.ents]

    # Return all NLP features
    return {
        'tokens': tokens,
        'lemmas': lemmas,
        'pos': pos_tags,
        'entities': entities
    }

# Read and process the CSV file
file_path = 'state_union_addresses.csv'

dictionary = defaultdict(lambda: {'TotalDocsFreq': 0, 'TotalCollectionFreq': 0})
postings = defaultdict(lambda: defaultdict(int))

# Modified document processing code
with open(file_path, mode='r', encoding='utf-8', errors='ignore') as infile:
    reader = csv.DictReader(infile)
    
    for row_num, row in enumerate(reader, start=1):
        speech_content = row['Text']
        president_name = row.get('President', '').replace(" ", "-")
        if president_name.startswith("-"):
            president_name = president_name[1:]
        
        date = row.get('Date', '')
        doc_id = f"{president_name}-{date}"
        #debugging print
        print(f"Processing {doc_id}, Speech Content (first 100 chars): {speech_content[:100]}")
        
        # Process text through NLP pipeline
        nlp_results = nlp_pipeline(speech_content)
        
       # Use lemmas for indexing
        terms = nlp_results['lemmas'] # Already cleaned by the pipeline
        term_freq = defaultdict(int)

        """ Define a stricter pattern for valid index terms
        Allows letters, numbers, and internal hyphens/apostrophes
        Must contain at least one letter or number
        Allow standard terms OR things looking like simple fractions """
        valid_index_term_pattern = re.compile(r"^(?:(?=.*[a-z0-9])[a-z0-9]+(?:[-'][a-z0-9]+)*|\d+/\d+)$")

        for term in terms:
            # Final validation before adding to index
            if valid_index_term_pattern.match(term):
                 term_freq[term] += 1
                 # Update collection frequency here as well
                 dictionary[term]['TotalCollectionFreq'] += 1 # Moved this here

        # Update document frequencies (postings)
        for term, freq in term_freq.items():
            # No need to check pattern again here, already filtered
            postings[term][doc_id] = freq
            dictionary[term]['TotalDocsFreq'] += 1 # Keep this update here

# Insert data into MySQL tables
for term, freqs in dictionary.items():
    cursor.execute(
        "INSERT IGNORE INTO Dictionary (Term, TotalDocsFreq, TotalCollectionFreq) VALUES (%s, %s, %s)",
        (term, freqs['TotalDocsFreq'], freqs['TotalCollectionFreq'])
    )

for term, docs in postings.items():
    for doc_id, freq in docs.items():
        cursor.execute(
            "INSERT IGNORE INTO Posting (Term, DocID, Term_Freq) VALUES (%s, %s, %s)",
            (term, doc_id, freq)
        )

print("\nCalculating TF-IDF vectors and Cosine Similarity Matrix...")

# New for part 2: Calculate Cosine Similarity
# --- Cosine Similarity Calculation ---

# 1. Fetch necessary data from DB
cursor.execute("SELECT Term, TotalDocsFreq FROM Dictionary")
dictionary_data = {row['Term']: row['TotalDocsFreq'] for row in cursor.fetchall()}

cursor.execute("SELECT Term, DocID, Term_Freq FROM Posting")
postings_data = cursor.fetchall()

# 2. Get all unique DocIDs and total number of documents (N)
all_doc_ids = sorted(list(set(p['DocID'] for p in postings_data)))
N = len(set(all_doc_ids_collected_during_indexing)))
if N == 0:
    print("No documents found in the Posting table. Cannot calculate similarity.")
    db.commit()
    cursor.close()
    db.close()
    exit() # Or handle appropriately

# 3. Pre-calculate IDF for all terms
idf_values = {}
for term, df in dictionary_data.items():
    if df > 0:
        idf_values[term] = math.log10(N / df)
    else:
        idf_values[term] = 0 # Should not happen if df is correct

# 4. Build TF-IDF vectors for each document and calculate vector magnitudes
doc_vectors = defaultdict(dict)
doc_magnitudes = defaultdict(float)

for posting in postings_data:
    term = posting['Term']
    doc_id = posting['DocID']
    tf = posting['Term_Freq']

    if term in idf_values:
        tf_idf = tf * idf_values[term]
        if tf_idf > 0: # Only store non-zero weights
            doc_vectors[doc_id][term] = tf_idf
            doc_magnitudes[doc_id] += tf_idf ** 2 # Sum of squares for magnitude

# Finalize magnitudes (square root)
for doc_id in doc_magnitudes:
    doc_magnitudes[doc_id] = math.sqrt(doc_magnitudes[doc_id])

# 5. Calculate Cosine Similarity Matrix
similarity_matrix = defaultdict(dict) 
processed_pairs = set() # Using a set, To avoid calculating (d1, d2) and (d2, d1)

print(f"Calculating similarities for {len(all_doc_ids)} documents...")
for i, doc_id1 in enumerate(all_doc_ids):
    if i % 10 == 0: # Progress indicator, for debug below
        print(f"  Processing document {i+1}/{N}...")
    for j, doc_id2 in enumerate(all_doc_ids):
        # Optimization: Only calculate for i < j to avoid duplicates and self-similarity
        if i >= j:
            if i == j:
                similarity_matrix[doc_id1][doc_id2] = 1.0 # Similarity with self is 1
            continue # Skip lower triangle and diagonal (already handled or symmetric)

        # Get vectors and magnitudes
        vector1 = doc_vectors[doc_id1]
        vector2 = doc_vectors[doc_id2]
        mag1 = doc_magnitudes[doc_id1]
        mag2 = doc_magnitudes[doc_id2]

        # Calculate dot product
        dot_product = 0.0
        # Iterate over terms in the smaller vector for efficiency
        if len(vector1) < len(vector2):
            for term, weight1 in vector1.items():
                if term in vector2:
                    dot_product += weight1 * vector2[term]
        else:
            for term, weight2 in vector2.items():
                if term in vector1:
                    dot_product += vector1[term] * weight2

        # Calculate cosine similarity
        if mag1 > 0 and mag2 > 0:
            cosine_sim = dot_product / (mag1 * mag2)
        else:
            cosine_sim = 0.0 # Handle cases where one vector is all zeros

        similarity_matrix[doc_id1][doc_id2] = cosine_sim
        # Store symmetric value
        similarity_matrix[doc_id2][doc_id1] = cosine_sim

print("Similarity matrix calculation complete.")

# Example: Save to CSV (for manual checking)
with open('similarity_matrix.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    header = ['DocID1', 'DocID2', 'Similarity']
    writer.writerow(header)
    for doc1 in similarity_matrix:
        for doc2 in similarity_matrix[doc1]:
            if doc1 <= doc2: # Write only upper triangle + diagonal
                writer.writerow([doc1, doc2, similarity_matrix[doc1][doc2]])

# Insert similarity data into the database table
print("\nInserting similarity matrix into the database...")
similarity_insert_sql = """
INSERT IGNORE INTO Similarity (DocID1, DocID2, Similarity)
VALUES (%s, %s, %s)
"""
insert_count = 0
for doc1 in similarity_matrix:
    for doc2 in similarity_matrix[doc1]:
        # Insert only the upper triangle + diagonal to match the primary key constraint
        # and avoid storing redundant symmetric pairs.
        # Ensure consistent ordering for the primary key (e.g., doc1 <= doc2).
        if doc1 <= doc2:
            try:
                cursor.execute(similarity_insert_sql, (doc1, doc2, similarity_matrix[doc1][doc2]))
                insert_count += cursor.rowcount # Count successful inserts
            except pymysql.Error as e:
                print(f"Error inserting similarity for ({doc1}, {doc2}): {e}")

print(f"Inserted {insert_count} similarity pairs into the database.") #debugging print

# --- End of Cosine Similarity Calculation ---

db.commit()
cursor.close()
db.close()
print("\nDatabase connection closed.")