import anthropic
import google.generativeai as genai
import replicate
import os
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
import concurrent.futures
import streamlit as st

# Load the environment variables
load_dotenv()

# Set up the API keys
replicate.api_token = os.getenv('REPLICATE_API_TOKEN')
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
anthropicKey = os.getenv('CLAUDE_API_KEY')
openai_api_key = os.getenv('OPENAI_API_KEY')

# Terminal Colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'
RED = '\033[91m'
GOLD = '\033[38;2;255;215;0m'

def gpt4o(prompt, systemMessage, openai_api_key):
    """Queries the GPT-4o model with the given prompt and system message."""
    client = OpenAI(api_key=openai_api_key)
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": systemMessage},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content

def claude(prompt, systemMessage=""):
    """Queries the Claude model with the given prompt and optional system message."""
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
    """Queries the Gemini model with the given prompt."""
    model = genai.GenerativeModel(model_name="gemini-1.0-pro")
    convo = model.start_chat(history=[])
    convo.send_message(prompt)
    return convo.last.text

def llama3(prompt):
    """Queries the Llama3 model with the given prompt and returns the result."""
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
    """Queries the MistralAI model with the given prompt and returns the result."""
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

def theKing(prompt, openai_api_key):
    """Orchestrates querying multiple models and synthesizes their responses into a final answer."""
    system_message = """You are a wise and knowledgeable coder and problem solver king who provides thoughtful answers to questions.
    You have 3 advisors, who offer their insights to assist you.

    Consider their perspectives and advice, but ultimately provide your own well-reasoned response to the problem based on all context
    and advice. If you find their input helpful, feel free to acknowledge their contributions in your answer."""

    models = {
        "Llama3": llama3,
        "MistralAI": mistralai,
        "Gemini": gemini,
        "Claude": claude
    }

    answers = {}
    color_mapping = {
        'Llama3': ':blue[Llama3]',
        'MistralAI': ':red[MistralAI]',
        'Gemini': ':violet[Gemini]',
        'Claude': ':green[Claude]'
    }

    with st.spinner("The King is gathering advice from advisors..."):
        with tqdm(total=len(models), desc="Gathering insights from advisors", unit="task") as progress_bar:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures_to_model = {executor.submit(model_func, prompt): model_name for model_name, model_func in models.items()}
                for future in concurrent.futures.as_completed(futures_to_model):
                    model_name = futures_to_model[future]
                    try:
                        answer = future.result()
                        header = f"**{model_name}'s advice:**" 
                        colored_header = header.replace(model_name, color_mapping.get(model_name, model_name))  # Apply color to header only
                        content_text = f"{colored_header}\n\n{answer}"
                        st.session_state.messages.append({"role": "assistant", "content": content_text})
                        st.chat_message("assistant").write(content_text)
                    except Exception as exc:
                        error_message = f"{model_name} generated an exception: {exc}"
                        answers[model_name] = error_message
                        error_content_text = f"**{model_name} Error:**\n\n{error_message}"
                        st.session_state.messages.append({"role": "assistant", "content": error_content_text})
                        st.chat_message("assistant").write(error_content_text)
                    progress_bar.update()

    with st.spinner("The King is crafting his response..."):
        peasant_answers = "\n\n".join(f"{name}'s advice: {advice}" for name, advice in answers.items())
        king_prompt = f"{peasant_answers}\n\nProblem: {prompt}\n\nUse the insights from the advisors to create a step-by-step plan to solve the given Problem, then solve the problem your way. Also, include footnotes to the best advisor contributions."
        king_answer = gpt4o(king_prompt, system_message, openai_api_key)

    return answers, king_answer

def clear_chat():
    """Clears the chat history."""
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Streamlit UI
st.title("💬 King Architecture")
st.caption("🚀 A Streamlit chatbot powered by multiple AI models")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Enter your prompt:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    model_answers, king_answer = theKing(prompt, openai_api_key)

    king_content_text = f"**:orange[The King's answer:]**\n\n{king_answer}"
    st.session_state.messages.append({"role": "assistant", "content": king_content_text})
    st.chat_message("assistant").write(king_content_text)

st.button("Clear Chat", on_click=clear_chat)