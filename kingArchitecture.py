import anthropic
import google.generativeai as genai

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