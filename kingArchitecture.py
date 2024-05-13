import anthropic
import google.generativeai as genai
import replicate
import os
from dotenv import load_dotenv

# Load the environment variables
load_dotenv()

# Set up the API keys
replicate.api_token = os.getenv('REPLICATE_API_TOKEN')
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
anthropicKey = os.getenv('CLAUDE_API_KEY')


# Terminal Colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

# Run claude() to start a conversation with the Claude model
def claude():
    client = anthropic.Anthropic(api_key=anthropicKey)
    message = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=4096,
        temperature=0.0,
        system="Simulate three brilliant, logical experts collaboratively achieving a goal. Each one verbosely explains their thought process in real-time, considering the prior explanations of others and openly acknowledging mistakes. At each step, whenever possible, each expert refines and builds upon the thoughts of others, acknowledging their contributions. They continue until there is a definitive answer to the goal. For clarity, your entire response should be in a markdown table.",
        messages=[
            {"role": "user", "content": "The goal is to create The MoM project seeks to harness the collective intelligence of multiple large language models (LLMs) to enhance decision-making and problem-solving capabilities in complex scenarios. The integration of diverse AI model outputs could significantly improve the accuracy and depth of responses, presenting an opportunity to innovate in AI-driven decision frameworks."}
        ]
    )

    print(message.content)

claude()

# Run gemini() to start a conversation with the Gemini model
def gemini():
    model = genai.GenerativeModel(model_name="gemini-1.0-pro")

    convo = model.start_chat(history=[])

    convo.send_message("Hello, how are you today?")
    print(PINK, convo.last.text, RESET_COLOR)

# Run llama3() to start a conversation with the Llama model
def llama3():
    for event in replicate.stream(
        "meta/meta-llama-3-70b-instruct",
        input={
            "top_p": 0.9,
            "prompt": "Work through this problem step by step:\n\nQ: Sarah has 7 llamas. Her friend gives her 3 more trucks of llamas. Each truck has 5 llamas. How many llamas does Sarah have in total?",
            "max_tokens": 512,
            "min_tokens": 0,
            "temperature": 0.6,
            "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            "presence_penalty": 1.15,
            "frequency_penalty": 0.2,
        },
    ):
        print(str(event), end="")
        
# Run mistralai() to start a conversation with the Mistral model       
def mistralai():
    for event in replicate.stream(
        "mistralai/mistral-7b-instruct-v0.2",
        input={
            "top_p": 0.9,
            "prompt": "Give me 5 restaurants in Montclair area in New jersey for mothers day",
            "max_tokens": 512,
            "min_tokens": 0,
            "temperature": 0.6,
            "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            "presence_penalty": 1.15,
            "frequency_penalty": 0.2,
        },
    ):
        print(str(event), end="")