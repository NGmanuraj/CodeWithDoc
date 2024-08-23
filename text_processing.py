from langchain.text_splitter import RecursiveCharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = [doc['text'] for doc in documents]

    # Parallelize text splitting
    with ThreadPoolExecutor() as executor:
        chunks = list(executor.map(lambda text: text_splitter.split_text(text), texts))
    
    # Flatten the list of chunks
    chunks = [chunk for sublist in chunks for chunk in sublist]
    
    return [{'text': chunk} for chunk in chunks]
