# MoM

The King AI application leverages multiple large language models (LLMs) to enhance decision-making by using a hierarchical approach where a central 'King' AI delegates tasks to 'Peasant' AIs, gathers their insights, evaluates them, and provides a final answer.

## Technology Stack

``` mermaid
graph TD;
    A[Backend] --> B[Python]
    B --> C[FastAPI]
    B --> D[AI Models]
    D --> E[Anthropic Claude]
    D --> F[Google Gemini]
    D --> G[Meta LLaMA]
    D --> H[Mistral AI]
    B --> I[APIs]
    I --> J[OpenAI API]
    I --> K[Replicate API]
    B --> L[Environment Management]
    L --> M[Python-dotenv]

```
### Technology	Description

**Python** -	Programming language used for backend and AI integration.
**FastAPI	Web** -  framework for building APIs with Python.
**Anthropic Claude** -	One of the LLMs used for generating insights.
**Google Gemini** -	Another LLM used for generating insights.
**Meta LLaMA** - An LLM used for generating insights.
**Mistral AI** -	An LLM used for generating insights.
**OpenAI API** -	API for accessing OpenAI's GPT models.
**Replicate API**	- API for accessing various AI models and tools.
**Python-dotenv** - Read key-value pairs from a .env file and set them as environment variables.

