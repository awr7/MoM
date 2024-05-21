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

# Terminal Colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'
RED = '\033[91m'
GOLD = '\033[38;2;255;215;0m'

def gpt4o(prompt, systemMessage):
    client = OpenAI()

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": systemMessage},
            {"role": "user", "content": prompt}
        ]
    )

    return completion.choices[0].message.content

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

    return message.content[0].text

def gemini(prompt):
    model = genai.GenerativeModel(model_name="gemini-1.0-pro")
    convo = model.start_chat(history=[])
    convo.send_message(prompt)
    return convo.last.text

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
            "prompt_template": "system\n\nYou are a helpful assistant\nuser\n\n{prompt}\nassistant\n\n",
            "presence_penalty": 1.15,
            "frequency_penalty": 0.2,
        },
    ):
        results.append(str(event))
    return " ".join(results)

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
            "prompt_template": "system\n\nYou are a helpful assistant\nuser\n\n{prompt}\nassistant\n\n",
            "presence_penalty": 1.15,
            "frequency_penalty": 0.2,
        },
    ):
        results.append(str(event))
    return " ".join(results)

def theKing(prompt):
    system_message = """You are a wise and knowledgeable coder and problem solver king who provides thoughtful answers to questions.
    You have 3 advisors, who offer their insights to assist you.

    Consider their perspectives and advice, but ultimately provide your own well-reasoned response to the problem based on all context
    and advice. If you find their input helpful, feel free to acknowledge their contributions in your answer."""

    models = {
        "llama3": llama3,
        "mistralai": mistralai,
        "gemini": gemini,
        "claude": claude
    }

    answers = {}
    color_mapping = {
        'llama3': PINK,
        'mistralai': CYAN,
        'gemini': YELLOW,
        'claude': NEON_GREEN,
        'gpt4o': GOLD
    }

    with tqdm(total=len(models), desc="Gathering insights from advisors", unit="task") as progress_bar:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures_to_model = {executor.submit(model_func, prompt): model_name for model_name, model_func in models.items()}
            for future in concurrent.futures.as_completed(futures_to_model):
                model_name = futures_to_model[future]
                try:
                    answers[model_name] = future.result()
                    color = color_mapping.get(model_name, RESET_COLOR)
                    print(f"\n{color}{model_name}'s advice:{RESET_COLOR}\n{answers[model_name]}\n")
                except Exception as exc:
                    answers[model_name] = f"{model_name} generated an exception: {exc}"
                    print(f"{RED}{model_name} generated an exception: {exc}{RESET_COLOR}")
                progress_bar.update()

    peasant_answers = "\n\n".join(f"{name}'s advice: {advice}" for name, advice in answers.items())
    king_prompt = f"{peasant_answers}\n\nProblem: {prompt}\n\nUse the insights from the advisors to create a step-by-step plan to solve the given Problem, then solve the problem your way. Also, include footnotes to the best advisor contributions."
    king_answer = gpt4o(king_prompt, system_message)

    return king_answer

def main():
    user_prompt = ""
    print("Please enter your prompt (type 'END' on a new line to finish):")
    while True:
        line = input()
        if line == "END":
            break
        user_prompt += line + "\n"
    final_answer = theKing(user_prompt)
    print("\nThe King's answer:\n")
    print(f"{GOLD}{final_answer}{RESET_COLOR}")

if __name__ == "__main__":
    main()
