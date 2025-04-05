import gradio as gr
import logging
from utils import get_agent

# Setup logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_agent(uploaded_pdf, state):
    """
    Create the agent from an uploaded PDF and store it in state.
    """
    try:
        agent = get_agent(uploaded_pdf)
        state = [agent]
        # Reveal the query input box once the agent is created
        return gr.update(visible=True, value="Ask a question"), state
    except Exception as e:
        logger.error(f"Error in create_agent: {e}")
        return gr.update(visible=False), state

def response_generator(query, state):
    """
    Process a user query using the stored agent and stream back the response.
    """
    try:
        logger.info(f"Received query: {query}")
        if not state or len(state) == 0:
            yield "Agent not initialized. Please upload a PDF first."
            return
        agent = state[0]
        response = agent.query(query)
        output = ""
        for text in response.response_gen:
            output += text
            yield gr.update(value=output)
    except Exception as e:
        logger.error(f"Error in response_generator: {e}")
        yield "An error occurred while processing your query."

def clear_conversation():
    """
    Clear the conversation by resetting input and state.
    """
    return gr.update(value=""), []

with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Agentic RAG Application

        Welcome to Agentic RAG – an intelligent PDF Q/A system powered by Llama Index and Gradio.
        Upload a PDF, then ask questions to retrieve information using our smart agent.
        """
    )
    
    state = gr.State([])

    with gr.Row():
        upload_button = gr.UploadButton("📁 Upload PDF", file_types=[".pdf"])
    
    input_box = gr.Textbox(autoscroll=True, visible=False, label="Your Query")
    output_box = gr.Textbox(autoscroll=True, max_lines=30, value="Output will appear here.", label="Agent Response")
    
    upload_button.upload(create_agent, [upload_button, state], [input_box, state], 
                           queue=False, show_progress=True, trigger_mode="once")
    input_box.submit(response_generator, [input_box, state], output_box)
    
    clear_btn = gr.Button("Clear Conversation")
    clear_btn.click(clear_conversation, None, [input_box, state])
    
demo.queue()
demo.launch(share=True)
