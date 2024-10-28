import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from textwrap import fill  # for printing wrapped text
import fitz  # for PDF handling

device = "cuda" if torch.cuda.is_available() else "cpu"
pdf_path = "rag.pdf"


def print_wrapped_text(text, wrap_length=80):
    """Prints text wrapped to a specified width."""
    print(fill(text, wrap_length))


def load_text_chunks(file_path: str) -> pd.DataFrame:
    """Loads text chunks and embeddings from a CSV file."""
    text_chunk_df = pd.read_csv(file_path)
    text_chunk_df["embedding"] = text_chunk_df["embedding"].apply(
        lambda x: np.fromstring(x.strip("[]"), sep=" ")
    )
    return text_chunk_df


def convert_embeddings_to_tensor(text_chunk_df: pd.DataFrame) -> torch.Tensor:
    """Converts embeddings from NumPy arrays to a torch tensor and sends them to the device."""
    embeddings = torch.tensor(
        text_chunk_df["embedding"].tolist(), dtype=torch.float32
    ).to(device)
    return embeddings


def retrieve_relevant_resources(
    query: str,
    embeddings: torch.Tensor,
    model: SentenceTransformer = SentenceTransformer(model_name_or_path="all-MiniLM-L6-v2", device=device),
    n_resources_to_return: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Embeds a query with the model and returns top k scores and indices from embeddings.
    """
    query_embedding = model.encode(query, convert_to_tensor=True)
    dot_scores = util.dot_score(query_embedding, embeddings)[0]
    scores, indices = torch.topk(input=dot_scores, k=n_resources_to_return)
    return scores, indices


def print_top_results_and_scores(
    query: str,
    embeddings: torch.Tensor,
    text_chunks: list[dict],
    n_resources_to_return: int = 5,
) -> None:
    """
    Takes a query, retrieves most relevant resources and prints them out in descending order.
    """
    scores, indices = retrieve_relevant_resources(
        query=query, embeddings=embeddings, n_resources_to_return=n_resources_to_return
    )

    print(f"Query: {query}\n")
    print("Results:")
    for score, index in zip(scores, indices):
        print(f"Score: {score:.4f}")
        print_wrapped_text(text_chunks[index]["sentence_chunk"])
        print(f"Page number: {text_chunks[index]['page_number']}")
        print("\n")

        # Optional: Open PDF and load target page (already shown in previous code)

        doc = fitz.open(pdf_path)
        page = doc.load_page(text_chunks[index]['page_number']) # number of page (our doc starts page numbers on page 41)

        # Get the image of the page
        img = page.get_pixmap(dpi=300)

        # Optional: save the image
        img.save(f"page_{text_chunks[index]['page_number']}.png")
        doc.close()


# # Load text chunks and embeddings
# text_chunk_df = load_text_chunks("text_chunks_and_embeddings_df.csv")
# text_chunks = text_chunk_df.to_dict(orient="records")
# embeddings = convert_embeddings_to_tensor(text_chunk_df)

# # Example usage
# query = input("Query")

