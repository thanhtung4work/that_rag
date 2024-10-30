import os

import re
import fitz
import pandas as pd
from sentence_transformers import SentenceTransformer
from spacy.lang.en import English
import torch
from tqdm.auto import tqdm


def open_and_read_pdf(pdf_path: str) -> list[dict]:
  """
  Opens a PDF file, reads its text content page by page, and collects statistics.

  Args:
      pdf_path (str): The file path to the PDF document to be opened and read.

  Returns:
      list[dict]: A list of dictionaries, each containing page statistics and text.
  """
  pages_and_texts = []

  doc = fitz.open(pdf_path)
  for page_number, page in tqdm(enumerate(doc)):
    text = page.get_text()  # get plain text encoded as UTF-8
    text = text_formatter(text)
    pages_and_texts.append({
      "document": pdf_path,
      "page_number": page_number,
      "page_char_count": len(text),
      "page_word_count": len(text.split(" ")),
      "page_sentence_count_raw": len(text.split(". ")),
      "page_token_count": len(text) / 4,  # 1 token = ~4 chars
      "text": text
    })
  return pages_and_texts


def open_and_read_directory(dir_path: str) -> list[dict]:
  document_pages_and_texts = []
  for f in os.listdir(dir_path):
    print("[INFO]: Reading ", f)
    pages_and_texts = open_and_read_pdf(f"{dir_path}/{f}")
    document_pages_and_texts.extend(pages_and_texts)
  return document_pages_and_texts


def text_formatter(text: str) -> str:
  """
  Performs minor formatting on text.

  Args:
      text (str): The text to be formatted.

  Returns:
      str: The formatted text.
  """
  return re.sub(r"\s+", " ", text).strip()


def extract_sentences(document: list) -> None:
  """
  Extracts sentences from each page text using spaCy.

  Args:
      document (list): A list of dictionaries containing page data.
  """
  nlp = English()
  nlp.add_pipe("sentencizer")
  for item in tqdm(document):
    sentences = list(nlp(item["text"]).sents)
    item["sentences"] = [str(sentence) for sentence in sentences]
    item["page_sentence_count_spacy"] = len(item["sentences"])


def chunk_sentences(document: list, chunk_size: int = 10) -> None:
  """
  Splits sentences into chunks of a specified size.

  Args:
      document (list): A list of dictionaries containing page data and sentences.
      chunk_size (int, optional): The size of each sentence chunk. Defaults to 10.
  """
  for item in tqdm(document):
    item["sentence_chunks"] = [item["sentences"][i:i + chunk_size] for i in range(0, len(item["sentences"]), chunk_size)]
    item["num_chunks"] = len(item["sentence_chunks"])


def join_chunk_sentences(document: list) -> list[dict]:
  """
  Joins sentences within each chunk into a single string (a paragraph).

  Args:
      document (list): A list of dictionaries containing page data and sentence chunks.

  Returns:
      list[dict]: A list of dictionaries containing chunk statistics and joined sentences.
  """
  pages_and_chunks = []
  for item in tqdm(document):
    for sentence_chunk in item["sentence_chunks"]:
      chunk_dict = {
        "document": item["document"],
        "page_number": item["page_number"],
        "sentence_chunk": "".join(sentence_chunk).replace("  ", " ").strip()
      }
      chunk_dict["sentence_chunk"] = re.sub(r'\.([A-Z])', r'. \1', chunk_dict["sentence_chunk"])
      chunk_dict["chunk_char_count"] = len(chunk_dict["sentence_chunk"])
      chunk_dict["chunk_word_count"] = len(chunk_dict["sentence_chunk"].split(" "))
      chunk_dict["chunk_token_count"] = len(chunk_dict["sentence_chunk"]) / 4
      pages_and_chunks.append(chunk_dict)
  return pages_and_chunks


def create_embeddings(document: list, embedding_model: SentenceTransformer) -> None:
  """
  Creates embeddings for each sentence chunk.

  Args:
      document (list): A list of dictionaries containing chunk data.
      embedding_model (SentenceTransformer): The embedding model to use.
  """
  for item in tqdm(document):
    item["embedding"] = embedding_model.encode(item["sentence_chunk"])


def save_embeddings(document: list, file_path: str) -> None:
  """
  Saves the embeddings to a CSV file.

  Args:
      document (list): A list of dictionaries containing chunk data and embeddings.
      file_path (str): The file path to save the embeddings to.
  """
  text_chunks_and_embeddings_df = pd.DataFrame(document)
  text_chunks_and_embeddings_df.to_csv(file_path, index=False)


# Main execution
if __name__ == "__main__":
  dir_path = "documents/"
  embeddings_df_save_path = "data/text_chunks_and_embeddings_df.csv"

  pages_and_texts = open_and_read_directory(dir_path)
  extract_sentences(pages_and_texts)
  chunk_sentences(pages_and_texts)
  pages_and_chunks = join_chunk_sentences(pages_and_texts)

  embedding_model = SentenceTransformer(
    model_name_or_path="all-MiniLM-L6-v2",
    device="cuda" if torch.cuda.is_available() else "cpu"
  )
  create_embeddings(pages_and_chunks, embedding_model)
  save_embeddings(pages_and_chunks, embeddings_df_save_path)