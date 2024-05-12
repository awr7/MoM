import anthropic
import google.generativeai as genai
import replicate
import openai

genai.configure(api_key="")
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()
    
def claude():
    client = anthropic.Anthropic(
        # defaults to os.environ.get("ANTHROPIC_API_KEY")
        api_key="",
    )
    message = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1000,
        temperature=0.0,
        system="Respond only in Yoda-speak.",
        messages=[
            {"role": "user", "content": "How are you today?"}
        ]
    )

    print(message.content)

def gemini():
    model = genai.GenerativeModel(model_name="gemini-1.0-pro")

    convo = model.start_chat(history=[])

    convo.send_message("Hello, how are you today?")
    print(convo.last.text)

# The meta/meta-llama-3-70b-instruct model can stream output as it's running.
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
        "frequency_penalty": 0.2
    },
):
    print(str(event), end="")