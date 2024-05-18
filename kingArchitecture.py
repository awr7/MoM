import anthropic
import google.generativeai as genai
import replicate
import os
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
import concurrent.futures

# Load the environment variables
load_dotenv()

# Set up the API keys
replicate.api_token = os.getenv('REPLICATE_API_TOKEN')
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
anthropicKey = os.getenv('CLAUDE_API_KEY')
OpenAI.api_key = os.getenv('OPENAI_API_KEY')

def gpt4o(prompt, systemMessage):

    client = OpenAI()

    #OpenAI
    completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": systemMessage },
        {"role": "user", "content": prompt}
    ]
    )

    return completion.choices[0].message


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
def claude(prompt, systemMessage=""):
    client = anthropic.Anthropic(api_key=anthropicKey)
    message = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=4096,
        temperature=0.0,
        system=systemMessage,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return message.content


# Run gemini() to start a conversation with the Gemini model
def gemini(prompt):
    model = genai.GenerativeModel(model_name="gemini-1.0-pro")

    convo = model.start_chat(history=[])

    convo.send_message(prompt)
    return convo.last.text

# Run llama3() to start a conversation with the Llama model
def llama3(prompt):
    results = []
    for event in replicate.stream(
        "meta/meta-llama-3-70b-instruct",
        input={
            "top_p": 0.9,
            "prompt": prompt,
            "max_tokens": 512,
            "min_tokens": 0,
            "temperature": 0.6,
            "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            "presence_penalty": 1.15,
            "frequency_penalty": 0.2,
        },
    ):
        results.append(str(event))
    return results
        
# Run mistralai() to start a conversation with the Mistral model       
def mistralai(prompt):
    results = []
    for event in replicate.stream(
        "mistralai/mistral-7b-instruct-v0.2",
        input={
            "top_p": 0.9,
            "prompt": prompt,
            "max_tokens": 512,
            "min_tokens": 0,
            "temperature": 0.6,
            "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            "presence_penalty": 1.15,
            "frequency_penalty": 0.2,
        },
    ):
        results.append(str(event))
    return results

def theKing(prompt):
    system_message = """You are a wise and knowledgeable coder and problem solver king who provides thoughtful answers to questions.
    You have 3 advisors, who offer their insights to assist you.

    Consider their perspectives and advice, but ultimatly provide your own well-reasoned response to the problem based on all context
    and advice. If you find their input helpful, feel free to acknowledge their contributions in your answer."""

    models = {
        "llama3": llama3,
        "mistralai": mistralai,
        "gemini": gemini,
        "claude": claude
    }

    answers = {}
    with tqdm(total=len(models), desc="Gathering insights from advisors", unit="task") as progress_bar:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures_to_model = {executor.submit(model_func, prompt): model_name for model_name, model_func in models.items()}
            for future in concurrent.futures.as_completed(futures_to_model):
                model_name = futures_to_model[future]
                try:
                    answers[model_name] = future.result()
                except Exception as exc:
                    answers[model_name] = f"{model_name} generated an exception: {exc}"
                progress_bar.update()

    peasant_answers = "\n\n".join(f"{name}'s advice: {advice}" for name, advice in answers.items())
    king_prompt = f"{peasant_answers}\n\nProblem: {prompt}\n\nUse the insights from the advisors to create a step-by-step plan to solve the given Problem, then solve the problem your way. Also, include footnotes to the best advisor contributions."
    king_answer = gpt4o(king_prompt, system_message)

    return king_answer

def main():
    user_prompt = input("Please enter your prompt: ")
    final_answer = theKing(user_prompt)
    print("\nThe King's answer:\n")
    print(final_answer)

if __name__ == "__main__":
    main()
