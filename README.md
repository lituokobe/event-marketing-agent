# Intelligent Customer Service Chatbot -- Creation Engine

This project provides an engine that allows users to freely design a
conversation flow based on nodes and conditional edges, and then build
an intelligent customer-service chatbot system powered by large language
models (LLMs).

The generated system can recognize user intents in real time, respond
accordingly, and effectively handle customer service tasks and
business-promotion scenarios.

## Features

-   **Custom Conversation Flow**: Freely orchestrate dialogue logic
    using normal nodes, transition nodes, and conditional edges
    according to business needs.
-   **Custom Intent Library**: Define keywords, question patterns, and
    LLM-based descriptions to detect user intent. These intents can be
    used when building normal nodes.
-   **Custom Knowledge Base**: For global questions that users may ask
    at any point, the system uses knowledge-base intent detection. You
    can configure whether a node prioritizes its local intent library or
    the global knowledge base.
-   **Multi-strategy Intent Detection**: Combines keyword matching,
    semantic similarity, and LLM-based reasoning.
-   **Real-time Response**: Low-latency dialog processing suitable for
    telephone environments.
-   **Highly Configurable**: Supports custom intent libraries and
    response strategies.

## Technical Architecture

### Core Models

| Component | Model Used | Description |
|-----------|------------|-------------|
| **Embedding Model** | Qwen/Qwen3-Embedding-0.6B | Used for semantic vectorization and similarity calculation |
| **Large Language Model** | Qwen-Plus (API) | Used for complex intent recognition and response generation |

### Intent Recognition Methods

-   **LLM-based analysis**
-   **Keyword matching**
-   **Semantic similarity retrieval**

## Environment Requirements

-   **Python**: 3.8 or above (3.11 recommended)
-   **Dependencies**: See `requirements.txt`

## Quick Start

### 1. Environment Setup

``` bash
# Clone the project
git clone https://github.com/lituokobe/event-marketing-agent
cd customer-service-bot

# Install dependencies
pip install -r requirements.txt
```

### 2. API Key Configuration

1.  Register an [Alibaba Cloud](https://www.alibabacloud.com/) account and obtain an API Key.
2.  Rename `.env.example` to `.env`.
3.  Fill in your API Key in `.env`:

``` bash
ALI_API_KEY=your_aliyun_api_key
```

### 3. Project Data

Project data is stored in the `./data` directory:

-   `agent_data.json` --- Full configuration of the agent
-   `chatflow_design.json` --- Conversation flow configuration including
    main flow, nodes, and conditional edges. Users can fully customize
    this to build different customer-service bots
-   `dialogs.json` --- Preset dialog/response templates
-   `intent_data.json` --- Intent library configuration. Nodes will use
    specific intents according to project needs
-   `knowledge.json` --- Knowledge-base configuration for answering
    global user questions
-   `vector_db_collection.json` --- Vector DB configuration used for
    question-pattern retrieval

### 4. Run a Test

``` bash
python run_chatflow.py
```

## How It Works

### When LLM Mode Is Enabled

-   The system uses the LLM for intent classification.
-   You can set an LLM confidence threshold. When the LLM score is below
    this threshold (e.g., 3), the system falls back to the two non-LLM
    methods below.

### When LLM Mode Is Disabled

#### Layer 1: Keyword Retrieval

-   Uses an AC Automaton to quickly match predefined keywords.
-   When keywords match, the corresponding intent is returned
    immediately.

#### Layer 2: Question-pattern Understanding

-   Retrieves the most similar question pattern using vector similarity.
-   Cosine similarity threshold: ≥ 0.8
-   Uses a vector database (ChromaDB) for efficient retrieval.

