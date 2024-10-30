# llm_handler.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from utils.retrieval.retrieve import load_text_chunks, convert_embeddings_to_tensor, retrieve_relevant_resources

class LLMHandler:
    def __init__(self, model_id="microsoft/Phi-3.5-mini-instruct", device="cuda"):
        self.model_id = model_id
        self.device = device
        print(f"[INFO]: Using {self.model_id}")
        
        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, 
            device_map=self.device, 
            torch_dtype=torch.float16, 
            attn_implementation='eager'
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        
        # Initialize pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

        # Default generation arguments
        self.generation_args = {
            "max_new_tokens": 512,
            "return_full_text": False,
            # "do_sample": False,
        }

        # Initial system message
        self.messages = [{"role": "system", "content": "You are an AI assistant."}]
    
    def prompt_formatter(self, query: str, context_items: list[dict]) -> list[dict]:
        """
        Augments query with context items.
        """
        context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])

        # print(f"[DEBUG]: {context}")
        
        base_prompt = """Now use the following context items to answer the user query: {context}\n
        User query: {query}\n
        Answer:
        """
        base_prompt = base_prompt.format(context=context, query=query)
        
        return [{"role": "user", "content": base_prompt}]

    def generate_response(self, query: str, context_items: list[dict]) -> str:
        """
        Generates response based on query and context items.
        """
        prompt = self.prompt_formatter(query=query, context_items=context_items)
        self.messages += prompt
        output = self.pipe(self.messages, **self.generation_args)
        return output[0]['generated_text']


def get_relevant_context(query: str, text_chunk_file: str) -> list[dict]:
    """
    Retrieves context items relevant to the query.
    """
    text_chunk_df = load_text_chunks(text_chunk_file)
    text_chunks = text_chunk_df.to_dict(orient="records")
    embeddings = convert_embeddings_to_tensor(text_chunk_df)
    
    # Retrieve relevant resources
    scores, indices = retrieve_relevant_resources(query=query, embeddings=embeddings, n_resources_to_return=8)
    context_items = [text_chunks[i] for i in indices]
    
    # Attach scores to context items
    for i, item in enumerate(context_items):
        item["score"] = scores[i].cpu()  # return score to CPU if necessary
    
    return context_items


if __name__ == "__main__":
    llm_handler = LLMHandler()
    query = input("[User]: ")
    text_chunk_file = "data/text_chunks_and_embeddings_df.csv"
    
    # Retrieve relevant context
    context_items = get_relevant_context(query, text_chunk_file)
    
    # Generate and print response
    response = llm_handler.generate_response(query, context_items)
    print(f"\n[AI Assistant]:\n{response}")
