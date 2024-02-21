import os
import itertools
import numpy as np
from numpy.linalg import norm
from pathlib import Path
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

STOP_WORDS = set(
    stopwords.words("english")
)  # Load english stop words for preprocessing


def preprocess(filepath: str, query=False, duplicate=False):
    contents = ""
    word_count = 0

    with open(filepath, 'r', encoding='UTF-8') as f:
        contents = f.read().casefold()
        word_count = len(contents.strip().split())

    contents = word_tokenize(contents)

    # Convert content into a list while retaining input order and removing non-alnum and STOP_WORDS
    temp = []
    for word in contents:
        if word.isalnum() and (word not in STOP_WORDS):
            if duplicate==True and word not in temp: temp.append(word)
            else: temp.append(word)

    contents = temp

    return contents, word_count


def invert_structure(contents):
    inverted_index = {}
    for filename, content in contents.items():
        for word in content:
            if word not in inverted_index:
                inverted_index[word] = set()
                inverted_index[word].add(filename)
            else:
                inverted_index[word].add(filename)
    return inverted_index

# Takes corpus of docs and returns positional index of all the terms
def positional_index(contents: dict):
    pos_index = {}

    for filename, content in contents.items():
        position = 0
        for word in content:
            if word not in pos_index:
                pos_index[word] = {}
            if filename not in pos_index[word]:
                pos_index[word][filename] = []
            pos_index[word][filename].append(position)
            position += 1

    return pos_index


# Calculates term frequency of each term for all documents, using all 5 weighting schemes. Formats data in the form of:
# term: {filename: binary scheme, raw scheme, term frequency, log normalization scheme, double normalization scheme}
def term_frequency(word_count, pos_index):
    tf = {}
    # Use data from positional index dictionary 
    for word in pos_index:
        result = {}
        doc_positions = pos_index[word]; # format is -> {doc_id: [list of positions]}
        for filename in doc_positions:
            positions = doc_positions[filename]; # list of positions for doc_id   

            # Define max_tf for double normalization scheme calculation
            max_tf = max(len(pos_index[w][doc_id]) for w in pos_index for doc_id in pos_index[w])

            # If document does not contain word
            if filename not in pos_index[word]:
                result = 0
            else:
                # Calculate each term frequency weighting scheme for each doc
                result[filename] = [
                    1, # binary
                    len(positions), # raw
                    len(positions) / word_count[filename], # term frequency
                    np.log(1 + len(positions)), # log normalization
                    0.5 + 0.5 * (len(positions) / max_tf) # double normalization
                ]
        tf[word] = result
    return tf


# Calculates the inverse document frequency
def inverse_doc_freq(contents, pos_index):
    idf = {}
    document_count = {}
    corpus_total = len(contents)

    # Check how many documents the word appears in
    for word in pos_index:
        document_count = len(pos_index[word])

        # Calculate the inverse document frequency for the word
        idf[word] = np.log((corpus_total)/(1+document_count))

    return idf


# Creates TF-IDF matrix for each weighting scheme
# def tf_idf_matrix(tf, idf, ):
#     tfidf = []
#     # tf = 'hello': {'doc_1.txt': 0.5348924583920}, {'doc_2.txt': 0.28859385611}
#     # idf = 'hello': 0.248918501321

#     # For each term frequency weighting scheme, calculate TF-IDF weightings
#     for i in range(0, 5):
#         tfidf.append({})
#         for word in tf:
#             tfidf[i][word] = {}
#             for filename in tf[word]:
#                 tfidf[i][word][filename] = tf[word][filename][i]*idf[word] # Insert into TF-IDF matrix
    
#     return tfidf

def tf_idf_matrix(tf, idf, query):
    tfidf = []
    query_vector = [0] * len(tf)  # Initialize query vector with zeros
    
    # Tokenize the query into words
    query_words = query.split()
    
    # Calculate term frequency for each word in the query
    query_term_frequency = {}
    for word in query_words:
        query_term_frequency[word] = query_words.count(word)
    
    # For each term frequency weighting scheme, calculate TF-IDF weightings
    for i in range(0, 5):
        tfidf_scheme = {}
        for word in tf:
            tfidf_scheme[word] = {}
            for filename in tf[word]:
                tfidf_value = tf[word][filename][i] * idf[word]  # Calculate TF-IDF value for documents
                tfidf_scheme[word][filename] = tfidf_value  # Insert into TF-IDF matrix
            
            # Calculate TF-IDF value for the query
            if word in query_term_frequency:
                query_value = query_term_frequency[word] * idf[word]
                query_vector[i] = query_value
        
        tfidf.append(tfidf_scheme)
    
    return tfidf, query_vector



# Calculates the cosine simularity for the query using each TF weighting scheme
import numpy as np

def cosine_sim(tfidf, query_vector):
    top_documents = {}

    for i, tfidf_scheme in enumerate(tfidf):
        similarity_scores = {}

        for filename, document_vector in tfidf_scheme.items():
            doc_vector = list(document_vector.values())  # Extract the document vector from the dictionary values
            dot_product = np.dot(query_vector[i], doc_vector)
            query_norm = np.linalg.norm(query_vector[i])
            document_norm = np.linalg.norm(doc_vector)

            if query_norm == 0 or document_norm == 0:
                cosine_similarity = 0
            else:
                cosine_similarity = dot_product / (query_norm * document_norm)

            similarity_scores[filename] = cosine_similarity

        sorted_scores = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)
        top_documents[i] = [(filename, score) for filename, score in sorted_scores[:5]]

    return top_documents





class QueryEngine:
    def __init__(self, path: str, debug=True, positional=False):
        # Load contents of the files from the data folder to the contents dict {filename: [W1, W2,..]}
        self.contents = {} # corpus of docs with preprocessed terms
        self.word_counts = {} # word count for each doc
        self.filenames = [] # names of each doc
        self.variants = ["binary", "raw", "tf", "log", "double_norm"] # different TF weighting schemes
        self.tf = {} # term frequencies of each doc
        self.idf = {} # inverse document frequencies for each word
        

        for filename in os.listdir(path):
            try:
                content, word_count = preprocess(path + "/" + filename, duplicate=positional) # create list of terms from each doc
            except Exception as e:
                if debug == True:
                    print(f"Failed to preprocess: {path+'/'+filename}\n\t", e)
            else:
                self.contents[filename] = content
                self.word_counts[filename] = word_count
                self.filenames.append(filename)
        
        #print(self.contents)
        self.index = positional_index(self.contents) if positional else invert_structure(self.contents)
        self.words = self.index.keys()
        self.tf = term_frequency(self.word_counts,self.index)
        self.idf = inverse_doc_freq(self.contents, self.index)

        #self.tfidf = tf_idf_matrix(self.tf, self.idf, self.query)

        # self.freq = { # HS NOTE: Not really sure what this is being used for tbh, someone else pls check
        #     word: len(postings) for word, postings in self.index.items()
        # }  # NOTE: Might be smart to sort this based on the freq
        return

    def _or(self, x: set, y: set) -> set:
        return x.union(y)

    def _and(self, x: set, y: set) -> set:
        x, y = list(x), list(y)
        x = sorted(x)
        y = sorted(y)
        answer = set()
        p1 = 0
        p2 = 0
        comparison = 0
        while p1 < len(x) and p2 < len(y):
            comparison += 1
            if x[p1] == y[p2]:
                answer.add(x[p1])
                p1 += 1
                p2 += 1
            else:
                comparison += 1
                if x[p1] < y[p2]:
                    p1 += 1
                else:
                    p2 += 1
        return answer, comparison  # x.intersection(y)

    def _not(self, term: str) -> set:
        negated = set()
        postings = self.index[term]
        for filename in self.filenames:
            if filename not in postings:
                negated.add(filename)
        return negated

    def process(self, query_sentence: str, query_operation_sequence: str) -> set:
        # Preprocess query and split query operations
        query_operation_sequence = query_operation_sequence.split(
            ", "
        )  # NOTE: Handle []'s later, taking input with ', ' as delim and spliting for now
        query_sentence = preprocess(query_sentence, query=True)
        # Pair query sentence with operations (for displaying to the user in the terminal)
        preprocessed_query = [
            item
            for pair in itertools.zip_longest(
                query_sentence, query_operation_sequence, fillvalue=""
            )
            for item in pair
            if item
        ]
        print(f"Preprocessed Query: {' '.join(preprocessed_query)}\n")

        # Core processing loop 
        result = self.index[query_sentence[0]]
        comparison = 0
        for i in range(1, len(query_sentence)):
            operation = query_operation_sequence[i - 1]
            if operation == "AND NOT":
                # t1 AND NOT t2 -> t1 AND (NOT t2) -> t1 AND (res: NOT t2)
                term = self._not(query_sentence[i])
                result, temp_comparison = self._and(result, term)
                comparison += temp_comparison
            elif operation == "OR NOT":
                # t1 OR NOT t2 -> t1 OR (NOT t2) -> t1 OR (res: NOT t2)
                term = self._not(query_sentence[i])
                result = self._or(result, term)
            else:
                term = self.index[query_sentence[i]]
                if operation == "AND":
                    result, temp_comparison = self._and(result, term)
                    comparison += temp_comparison
                elif operation == "OR":
                    result = self._or(result, term)
                else:
                    print("Invalid Operation.")
                    exit()
        return result, comparison
