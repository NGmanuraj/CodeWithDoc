from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def create_faiss_index(chunks, batch_size=32):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Encode the text chunks in batches
    embeddings = []
    with ThreadPoolExecutor() as executor:
        futures = []
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i+batch_size]
            future = executor.submit(model.encode, [chunk['text'] for chunk in batch_chunks])
            futures.append(future)
        
        for future in futures:
            embeddings.extend(future.result())

    embeddings = np.array(embeddings).astype('float32')

    # Create a FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Store the texts and index in a dictionary
    vectorstore = {
        'index': index,
        'texts': [chunk['text'] for chunk in chunks]
    }
    
    return vectorstore
