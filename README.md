# Agentic RAG - PDF Q/A System

Agentic RAG is a production-ready application that leverages the power of Llama Index, Ollama, and Gradio to provide an intelligent question-answering system over PDF documents. The application processes PDFs, constructs an intelligent agent that uses both summarization and vector-based Q/A pipelines, and provides a clean, interactive interface for querying document content.

![alt](https://github.com/user-attachments/assets/a2859bd4-28a1-4636-b07f-ad448e232450)

## Repository Structure

```
.
├── config.py             # Central configuration for models and processing parameters
├── logging_config.py     # Centralized logging configuration
├── utils.py              # Core functions for PDF processing and agent creation
├── app.py                # Gradio-based interactive UI
├── requirements.txt      # Python dependencies
├── Makefile              # Optional commands for installation and setup
└── README.md             # Project documentation
```

## Getting Started

### Prerequisites

- **Python 3.8+**
- **Ollama Server:** Ensure the Ollama server is installed and running.
- **System Requirements:** Recommended Linux environment with at least 8GB RAM.

### Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/agentic-rag.git
   cd agentic-rag
   ```

2. **Create a Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Setup Using Makefile:**
   ```bash
   make install
   ```

## Running the Application

1. **Start the Ollama Server:**
   Ensure your Ollama server is running (this is required for LLM operations).

2. **Launch the Application:**
   ```bash
   python app.py
   ```

3. **Use the Interface:**
   A Gradio interface will launch in your browser. Upload a PDF and start asking questions!

## Dependencies

* llama-index==0.10.36
* llama-index-llms-ollama==0.1.3
* llama-index-embeddings-ollama==0.1.2
* pypdf==4.2.0
* gradio==4.31.2

## Makefile Commands

* `make install`: Install Python dependencies.
* `make models`: (Optional) Command to pull required models via Ollama.
* `make ollama_download`: (Optional) Instructions for downloading and starting the Ollama server.
