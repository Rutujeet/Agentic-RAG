import logging
from config import *
from llama_index.core import Settings, Document
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from pypdf import PdfReader
from llama_index.core import SummaryIndex, VectorStoreIndex, PromptTemplate
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector

logger = logging.getLogger(__name__)

# Initialize LLM and Embedding with our configured settings
Settings.llm = Ollama(model=LLM_MODEL, request_timeout=REQUEST_TIMEOUT, temperature=TEMPERATURE)
Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL)

def pre_processing(reader):
    """
    Extract text from a PDF reader, clean it, and split it into overlapping chunks.
    """
    try:
        full_text = ""
        for idx, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text(0)
            if page_text:
                full_text += page_text
            else:
                logger.warning(f"Page {idx} produced no text.")
        cleaned_text = full_text.replace('\n', ' ').lower()
        start_index = cleaned_text.find(FIRST_SECTION)
        end_index = cleaned_text.rfind(IGNORE_AFTER)
        if start_index != -1 and end_index != -1:
            cleaned_text = cleaned_text[start_index:end_index]
        sentences = cleaned_text.split('. ')
        chunks = []
        i = 0
        while i < len(sentences):
            chunk = '. '.join(sentences[i:i+GROUP_SIZE])
            if len(chunk) > 10:
                chunks.append(chunk)
            i += GROUP_SIZE - OVERLAP
        return chunks
    except Exception as e:
        logger.error(f"Error in pre_processing: {e}")
        raise

# Prompt Templates for different pipelines
qa_prompt_tmpl_str = (
    "<|user|>\n"
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, answer the query\n"
    "Query: {query_str}"
    " <|end|>\n"
    "<|assistant|>"
)

refine_prompt_tmpl_str = (
    "<|user|>\n"
    "The original query is as follows: {query_str}\n"
    "We have provided an existing answer: {existing_answer}\n"
    "We have the opportunity to refine the existing answer (only if needed) with some more context below.\n"
    "---------------------\n"
    "{context_msg}\n"
    "---------------------\n"
    "Given the new context, refine the original answer to better answer the query. If the context isn't useful, return the original answer."
    " <|end|>\n"
    "<|assistant|>"
)

summary_prompt_tmpl_str = (
    "<|user|>\n"
    "Context information from multiple sources is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the information from multiple sources and not prior knowledge, answer the query.\n"
    "Query: {query_str}"
    " <|end|>\n"
    "<|assistant|>"
)

def get_agent(uploaded_pdf):
    """
    Create a query agent by processing the uploaded PDF.
    Builds two pipelines (summarization and vector-based Q/A) and returns a RouterQueryEngine.
    """
    try:
        reader = PdfReader(uploaded_pdf)
        chunks = pre_processing(reader)
        if not chunks:
            raise ValueError("No valid text extracted from the PDF.")
        documents = [Document(text=chunk) for chunk in chunks]
        documents_summary = [Document(text=chunk) for chunk in chunks[:1]]
        
        summary_index = SummaryIndex(documents_summary)
        vector_index = VectorStoreIndex(documents)
        
        vector_query_engine = vector_index.as_query_engine(streaming=True)
        summary_query_engine = summary_index.as_query_engine(response_mode="tree_summarize", streaming=True)
        
        qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
        vector_query_engine.update_prompts({
            "response_synthesizer:text_qa_template": qa_prompt_tmpl
        })
        
        refine_prompt_tmpl = PromptTemplate(refine_prompt_tmpl_str)
        vector_query_engine.update_prompts({
            "response_synthesizer:refine_template": refine_prompt_tmpl
        })
        
        summary_prompt_tmpl = PromptTemplate(summary_prompt_tmpl_str)
        summary_query_engine.update_prompts({
            "response_synthesizer:summary_template": summary_prompt_tmpl
        })
        
        summary_tool = QueryEngineTool.from_defaults(
            query_engine=summary_query_engine,
            description="Useful for summarization or general description questions"
        )
        
        query_tool = QueryEngineTool.from_defaults(
            query_engine=vector_query_engine,
            description="Useful for specific questions or topic based questions not related to summarization"
        )
        
        agent = RouterQueryEngine(
            selector=LLMSingleSelector.from_defaults(),
            query_engine_tools=[summary_tool, query_tool],
            verbose=True
        )
        return agent
    except Exception as e:
        logger.error(f"Error in get_agent: {e}")
        raise