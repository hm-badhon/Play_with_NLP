import openai
import tiktoken

openai.api_key = "your_openai_api_key"  # Replace with your actual API key

def count_tokens(text, model="gpt-3.5-turbo"):
    """Counts the number of tokens in a given text."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def llm_based_chunking(text, max_tokens=1000):
    """Uses GPT to intelligently chunk text while preserving meaning."""
    
    # If the text is already within token limits, return it as a single chunk
    if count_tokens(text) <= max_tokens:
        return [text]

    prompt = f"""
    Your task is to split the following text into meaningful chunks while preserving context.
    Each chunk should not exceed {max_tokens} tokens.

    Text:
    {text}

    Return a JSON array of text chunks.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",  # Use gpt-3.5-turbo or gpt-4
        messages=[{"role": "system", "content": prompt}],
        temperature=0.5
    )

    chunks = response["choices"][0]["message"]["content"]
    return eval(chunks)  # Convert JSON output to Python list

# Example Long Text
document_text = """
Artificial Intelligence is revolutionizing industries. Machine Learning (ML) and Deep Learning are subsets of AI.
ML allows systems to learn from data, while Deep Learning uses neural networks for complex tasks.

AI applications include healthcare, finance, and robotics. In healthcare, AI assists in disease diagnosis and personalized medicine.
Financial institutions use AI for fraud detection and risk assessment.

Self-driving cars rely on AI for real-time decision-making, while chatbots enhance customer support.

Despite its benefits, AI raises ethical concerns. Issues such as bias, privacy, and automation-driven job losses must be addressed.
"""

# Apply LLM-Based Chunking
chunks = llm_based_chunking(document_text, max_tokens=100)

# Print Results
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:\n{chunk}\n{'-'*40}")
