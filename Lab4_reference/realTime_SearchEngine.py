import pymysql #type: ignore
import re
import math
from collections import defaultdict
import spacy  # type: ignore

# MySQL connection setup (same as in CSV_inverted_Index.py)
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "2842254K",
    "database": "BigDataLab4",
    "charset": "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor
}

"""Copied NLP pipeline from CSV_inverted_Index.py"""
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

    # 2. Apply lemmatization (with same improved cleaning)
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

def fetch_doc_count():
    """Get the total number of documents in the collection"""
    with pymysql.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(DISTINCT DocID) as doc_count FROM Posting")
            return cursor.fetchone()['doc_count']

def search_query(query_text, top_n=10):
    """Search for documents matching the query and return top N results"""
    nlp_result = nlp_pipeline(query_text)
    query_terms = nlp_result['lemmas']  # Use tokens for search
    if not query_terms:
        return []
    
    with pymysql.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cursor:
            # Get total document count for IDF calculation
            N = fetch_doc_count()
            
            # Dictionary to store document scores
            doc_scores = defaultdict(float)
            
            # Process each term in the query
            for term in query_terms:
                # Get the document frequency (number of docs containing this term)
                cursor.execute("SELECT TotalDocsFreq FROM Dictionary WHERE Term = %s", term)
                result = cursor.fetchone()
                
                if not result:
                    continue  # Term not found in any document
                
                df = result['TotalDocsFreq']
                idf = math.log10(N / df) if df > 0 else 0
                
                # Get all documents containing this term and their term frequencies
                cursor.execute(
                    "SELECT DocID, Term_Freq FROM Posting WHERE Term = %s", 
                    term
                )
                postings = cursor.fetchall()
                
                # Calculate TF-IDF score for each document
                for posting in postings:
                    doc_id = posting['DocID']
                    tf = posting['Term_Freq']
                    # Simple TF-IDF scoring
                    doc_scores[doc_id] += tf * idf
            
            # Sort documents by score in descending order
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Return the top N documents
            return sorted_docs[:top_n]

def display_results(results):
    """Format and display search results"""
    if not results:
        print("No matching documents found.")
        return
    
    print(f"\nTop {len(results)} most relevant documents:")
    print("-" * 60)
    
    for i, (doc_id, score) in enumerate(results, 1):
        # Extract document info from doc_id format: "Doc_Id: President-Date"
        doc_parts = doc_id.split(": ", 1)
        if len(doc_parts) > 1:
            # Just display the document info without further parsing
            doc_info = doc_parts[1]
            print(f"{i}. Document: {doc_info}")
            print(f"   Relevance Score: {score:.4f}")
            print("-" * 60)
        else:
            print(f"{i}. {doc_id} (Score: {score:.4f})")
            print("-" * 60)

def main():
    """Main function to run the search engine"""
    print("=" * 60)
    print("Welcome to the Presidential Speeches Search Engine")
    print("=" * 60)
    
    while True:
        query = input("\nEnter your search query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        
        results = search_query(query)
        display_results(results)

if __name__ == "__main__":
    main()
