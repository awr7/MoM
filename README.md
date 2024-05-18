# MoM

Harness the collective intelligence of multiple large language models (LLMs) to enhance decision-making and problem-solving capabilities in complex scenarios. 

## What do we hope to achieve with the integration of diverse AI model outputs?

- Synthesize responses from various LLMs
- To improve decision frameworks.
- Construct and rigorously test two collaborative architectures

## Technology Stack

```mermaid
graph TD;
    B[Python]
    B --> D[AI Models]
    D --> E[Claude Opus ]
    D --> F[Google Gemini]
    D --> K[Replicate API]
    K --> G[Meta LLaMA]
    K --> H[Mistral AI]
    D --> J[GPT 4o]
    B --> L[Environment Management]
    L --> M[Python-dotenv]

```

| Technology | Description |
| --- | --- |
| Python | Programming language used for backend and AI integration. |
| Anthropic Claude | One of the LLMs used for generating insights. |
| Google Gemini | Another LLM used for generating insights. |
| Meta LLaMA | An LLM used for generating insights. |
| Mistral AI | An LLM used for generating insights. |
| OpenAI API | API for accessing OpenAI's GPT models. |
| Replicate API | API for accessing various AI models and tools. |
| Python-dotenv | Read key-value pairs from a .env file and set them as environment variables. |


## King Data Flow

```mermaid
graph LR;
    A[User Input] --> B[Backend API]
    B --> C[Task Distribution]
    C --> D[Peasant AIs]
    D --> E[Meta LLaMA]
    D --> F[Google Gemini]
    D --> G[Mistral AI]
    D --> H[Anthropic Claude]
    E --> I[Responses]
    F --> I
    G --> I
    H --> I
    I --> J[King AI Evaluation]
    J --> K[Final Answer]
    K --> L[User Output]

```

## Duopoly Data Flow

```mermaid
graph LR
    
    B(User Input) --> C[GPT 4o]
    B --> D[Claude Opus]
    C --> E[Advisor Models Provide Insights]
    D --> E
    E --> F[Primary Models Discuss Findings]
    F --> G[Resolve Conflicts]
    G --> H[Reach Consensus]
    H --> I[Provide Final Answer]
    I --> J[End]

```
