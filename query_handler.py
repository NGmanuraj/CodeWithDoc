from sentence_transformers import SentenceTransformer
import numpy as np

def search_information(query, vectorstore):

    # Load the model (assuming it's already downloaded)
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate query embedding
    try:
        query_embedding = model.encode([query])
    except Exception as e:
        print(f"Error encoding query: {e}")
        return None  # Indicate error

    query_embedding = np.array(query_embedding).astype('float32')

    # Perform the search
    index = vectorstore['index']
    texts = vectorstore['texts']

    try:
        distances, indices = index.search(query_embedding, k=3)  # Get top 3 matches
    except Exception as e:
        print(f"Error performing search: {e}")
        return None  # Indicate error

    # Check for empty results
    if not distances.any():
        print("No relevant documents found for the query.")
        return None  # Indicate no relevant results

    # Retrieve and clean up the most relevant texts
    most_relevant_texts = texts[indices[0][0]].replace("\n", " ").strip()

    # Remove HTML tags and replace special characters
    cleaned_texts = most_relevant_texts.replace("&#160;", " ").replace("<p>", "").replace("</p>", "").replace("<span>", "").replace("</span>", "")

    # Combine the relevant texts into a single response
    combined_text = " ".join(cleaned_texts)

    # Format the combined text for better readability
    formatted_text = combined_text.replace(" ", " ").replace("&#160;", " ")  # Replace non-breaking spaces
    formatted_text = formatted_text.replace(",", ", ")  # Add spaces around commas
    formatted_text = formatted_text.replace("&#160;", " ")  # Remove any remaining non-breaking spaces

    # Split the text into paragraphs if needed
    paragraphs = formatted_text.split("\n\n")

    # Ensure the result is a single string
    paragraphs = str(paragraphs)

    # Return the formatted text as a list of paragraphs
    return paragraphs