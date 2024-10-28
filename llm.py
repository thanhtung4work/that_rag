from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from retrieve import load_text_chunks, convert_embeddings_to_tensor, retrieve_relevant_resources

model_id = "microsoft/Phi-3.5-mini-instruct"
print(f"[INFO]: Using {model_id}")

model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map="cuda", 
    torch_dtype="auto", 
    trust_remote_code=True, 
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
]

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "do_sample": False,
}

# output = pipe(messages, **generation_args)
# print(output[0]['generated_text'])

def prompt_formatter(query: str, context_items: list[dict]) -> list[dict]:
    """
    Augments query with text-based context from context_items.
    """
    # Join context items into one dotted paragraph
    context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])

    # Create a base prompt with examples to help the model
    # Note: this is very customizable, I've chosen to use 3 examples of the answer style we'd like.
    # We could also write this in a txt file and import it in if we wanted.
    base_prompt = """Now use the following context items to answer the user query:
    {context}\nRelevant passages: <extract relevant passages from the context here>
    User query: {query}
    Answer:
    """

    # Update base prompt with context items and query   
    base_prompt = base_prompt.format(context=context, query=query)

    # Create prompt template for instruction-tuned model
    dialogue_template = [
        {"role": "user", "content": base_prompt}
    ]

    
    return dialogue_template

text_chunk_df = load_text_chunks("text_chunks_and_embeddings_df.csv")
text_chunks = text_chunk_df.to_dict(orient="records")
embeddings = convert_embeddings_to_tensor(text_chunk_df)

# Example usage
query = input("Query: ")

# Get relevant resources
scores, indices = retrieve_relevant_resources(query=query, embeddings=embeddings)

# Create a list of context items
context_items = [text_chunks[i] for i in indices]

# Add score to context item
for i, item in enumerate(context_items):
    item["score"] = scores[i].cpu() # return score back to CPU 

# Format the prompt with context items
prompt = prompt_formatter(query=query, context_items=context_items)
messages += prompt
output = pipe(messages, **generation_args)

print(output[0]['generated_text'])
print(messages)