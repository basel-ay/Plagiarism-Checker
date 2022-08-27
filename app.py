import os
from pydoc import doc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# read all text files and get its content
documents = [doc for doc in os.listdir('documents') if doc.endswith('.txt')]
documents_contents = [open('documents/' + document,
                           encoding='utf-8').read() for document in documents]


def vectorize_text(text):
    """
    Apply TfidfVectorizer on a text
    """
    return TfidfVectorizer().fit_transform(text).toarray()


def get_similarity(doc1, doc2):
    """
    return cosine similarity between two documents
    """
    return cosine_similarity([doc1, doc2])


# Apply TfidfVectorizer
documents_vectors = vectorize_text(documents_contents)
# iterator of tuples where the first item in each passed iterator is paired together
zip_documents_with_vectors = list(zip(documents, documents_vectors))
plagiarism_results = set()


def check_plagiarism():
    """
    get similarity between each pair of documents and return similarity score
    """
    global zip_documents_with_vectors
    for document_a, text_vector_a in zip_documents_with_vectors:
        new_vectors = zip_documents_with_vectors.copy() # get a copy
        current_index = new_vectors.index((document_a, text_vector_a)) # get its index
        del new_vectors[current_index] # remove it from the copy to make comparison with rest documents

        for document_b, text_vector_b in new_vectors:
            sim_score = get_similarity(text_vector_a, text_vector_b)[0][1]
            document_pair = sorted((document_a, document_b))
            # print the two documents names and similarity between them
            score = (document_pair[0], document_pair[1], sim_score)
            plagiarism_results.add(score)  # all scores (unique elements)

    return plagiarism_results


for data in check_plagiarism():
    print(data)
