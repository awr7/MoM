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
anthropic_key = os.getenv('CLAUDE_API_KEY')
openai_api_key = os.getenv('OPENAI_API_KEY')

def gpt4o(prompt, system_message, openai_api_key):
    client = OpenAI(api_key=openai_api_key)
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content

def claude(prompt, system_message=""):
    client = anthropic.Anthropic(api_key=anthropic_key)
    message = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=4096,
        temperature=0.0,
        system=system_message,
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
            "temperature": 0.6,
            "prompt_template": "system\n\nYou are a helpful assistant\nuser\n\n{prompt}\nassistant\n\n",
            "presence_penalty": 1.15,
            "frequency_penalty": 0.2,
        },
    ):
        results.append(str(event))
    return " ".join(results)

# King architecture function
def the_king(prompt, openai_api_key):
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
    king_prompt = f"{peasant_answers}\n\nProblem: {prompt}\n\nUse the insights from the advisors to create a step-by-step plan to solve the given problem, then solve the problem your way. Also, include footnotes to the best advisor contributions."
    king_answer = gpt4o(king_prompt, system_message, openai_api_key)

    return answers, king_answer

def duopoly(prompt, openai_api_key):
    system_message_oi = f"You are a wise and knowledgeable OpenAI coder and problem solver expert who provides thoughtful answers to questions. Discuss and push back at Claude3 Oracle, challenge his suggestions and evaluate the best solutions based on the context from other advisors' answers to solve the problem: {prompt}"
    system_message_c3 = f"You are a wise and knowledgeable Claude3 coder and problem solver expert who provides thoughtful answers to questions. Discuss and push back at OpenAI Oracle, challenge his suggestions and evaluate the best solutions based on the context from other advisors' answers to solve the problem: {prompt}"
    system_message5 = "You are an expert at looking at a conversation between two smart oracles and extracting the best answer to a problem from the conversation."

    conversation_history = []

    # Initial advisors insights
    models = {
        "Llama3": llama3,
        "MistralAI": mistralai,
        "Gemini": gemini,
        "Claude": claude
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
    oracle_prompt = f"{peasant_answers}\n\nHello Oracle OpenAI, this is Oracle Claude3. Let's discuss and find a solution to the problem while challenging and taking the advisors' insights into consideration. Solve the problem: {prompt}"

    conversation_history.append(oracle_prompt)

    # Simulate conversation between OpenAI and Claude
    for i in range(1):
        current_context = "\n".join(conversation_history)
        if i % 2 == 0:
            claude_message = claude(current_context, system_message_c3)
            conversation_history.append(f"Oracle Claude3 said: {claude_message}\n")
        else:
            openai_message = gpt4o(current_context, system_message_oi, openai_api_key)
            conversation_history.append(f"Oracle OpenAI responded: {openai_message}\n")

    full_conversation = "\n".join(conversation_history)
    final_response = gpt4o(f"Summarize the conversation and conclude with a final answer to the problem under 100 words:\n{full_conversation}", system_message5, openai_api_key)
    return answers, final_response

# Streamlit UI
st.title("💬 Choose Your AI Architecture")
st.caption("🚀 A Streamlit chatbot powered by multiple AI models")

# Step 1: Choose the architecture
architecture_choice = st.selectbox("Choose the architecture", ["King", "Duopoly"])

# Step 2: Initialize messages in session state if not already present
st.session_state.setdefault("messages", [{"role": "assistant", "content": "How can I help you?"}])

# Step 3: Define custom avatars for each model
model_avatars = {
    "Llama3": "🦙",
    "MistralAI": "🌬️",
    "Gemini": "♊",
    "Claude": "🧠",
    "assistant": ":material/smart_toy:",
    "user": ":material/person:",
    "King": "👑" 
}

default_avatar = ":material/person:"

# Step 4: Display existing messages with appropriate avatars
for msg in st.session_state.messages:
    avatar = model_avatars.get(msg["role"], default_avatar)
    st.chat_message(msg["role"], avatar=avatar).write(msg["content"])

# Step 5: Input new prompt and display it
if prompt := st.chat_input("Enter your prompt:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user", avatar=model_avatars["user"]).write(prompt)

    # Step 6: Process the prompt based on chosen architecture
    with st.spinner(f"The {architecture_choice} is gathering advice from advisors..."):
        if architecture_choice == "King":
            model_answers, final_answer = the_king(prompt, openai_api_key)
        elif architecture_choice == "Duopoly":
            model_answers, final_answer = duopoly(prompt, openai_api_key)

    # Display each model's advice
    for model_name, answer in model_answers.items():
        st.session_state.messages.append({"role": "assistant", "content": f"{model_name}'s advice: {answer}"})
        st.chat_message("assistant", avatar=model_avatars.get(model_name, default_avatar)).write(f"{model_name}'s advice: {answer}")

    # Display the final answer from the chosen architecture
    st.session_state.messages.append({"role": "assistant", "content": f"The {architecture_choice}'s answer: {final_answer}"})
    st.chat_message("assistant", avatar=model_avatars["King"]).write(f"The {architecture_choice}'s answer: {final_answer}")
