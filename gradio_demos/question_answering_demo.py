"""
Gradio demo for Question Answering systems.
"""

import gradio as gr
from question_answering.model import ExtractiveQA, GenerativeQA, HybridQA
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize models
extractive_qa = ExtractiveQA()
generative_qa = GenerativeQA()
hybrid_qa = HybridQA()


def answer_extractive(question, context, top_k):
    """Extractive QA."""
    try:
        answers = extractive_qa.answer(question, context, top_k=int(top_k))
        result = ""
        for i, ans in enumerate(answers, 1):
            result += f"**Answer {i}:** {ans['answer']}\n"
            result += f"*Confidence:* {ans['score']:.4f}\n\n"
        return result
    except Exception as e:
        return f"Error: {str(e)}"


def answer_generative(question, context):
    """Generative QA."""
    try:
        answer = generative_qa.answer(question, context)
        return answer
    except Exception as e:
        return f"Error: {str(e)}"


def answer_hybrid(question, context, mode):
    """Hybrid QA."""
    try:
        result = hybrid_qa.answer(question, context, mode=mode.lower())
        response = f"**Answer:** {result['answer']}\n\n"
        response += f"**Method:** {result['method']}\n"
        if 'score' in result:
            response += f"**Confidence:** {result['score']:.4f}"
        return response
    except Exception as e:
        return f"Error: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="Question Answering") as demo:
    gr.Markdown("# ❓ Question Answering System")
    gr.Markdown("Answer questions using extractive, generative, or hybrid approaches")
    
    with gr.Tabs():
        # Extractive QA tab
        with gr.Tab("🔍 Extractive QA"):
            gr.Markdown("Find answer spans in the provided context")
            with gr.Row():
                with gr.Column():
                    ext_context = gr.Textbox(
                        label="Context",
                        placeholder="Enter context text...",
                        lines=8
                    )
                    ext_question = gr.Textbox(
                        label="Question",
                        placeholder="What do you want to know?",
                        lines=2
                    )
                    ext_topk = gr.Slider(1, 5, value=3, step=1, label="Number of Answers")
                    ext_button = gr.Button("Find Answer", variant="primary")
                with gr.Column():
                    ext_output = gr.Markdown(label="Answers")
            
            ext_button.click(
                answer_extractive,
                inputs=[ext_question, ext_context, ext_topk],
                outputs=ext_output
            )
            
            gr.Examples(
                examples=[
                    ["Where is the Eiffel Tower located?", "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel.", 3],
                    ["Who created Python?", "Python is a high-level programming language created by Guido van Rossum and first released in 1991.", 2],
                ],
                inputs=[ext_question, ext_context, ext_topk]
            )
        
        # Generative QA tab
        with gr.Tab("✨ Generative QA"):
            gr.Markdown("Generate answers from context or from knowledge")
            with gr.Row():
                with gr.Column():
                    gen_question = gr.Textbox(
                        label="Question",
                        placeholder="What do you want to know?",
                        lines=2
                    )
                    gen_context = gr.Textbox(
                        label="Context (optional for open-domain)",
                        placeholder="Enter context or leave empty...",
                        lines=8
                    )
                    gen_button = gr.Button("Generate Answer", variant="primary")
                with gr.Column():
                    gen_output = gr.Textbox(label="Answer", lines=5)
            
            gen_button.click(
                answer_generative,
                inputs=[gen_question, gen_context],
                outputs=gen_output
            )
            
            gr.Examples(
                examples=[
                    ["What is the capital of France?", ""],
                    ["Who wrote Romeo and Juliet?", ""],
                    ["What are the main features?", "Python is a high-level, interpreted programming language. It emphasizes code readability and supports multiple programming paradigms."],
                ],
                inputs=[gen_question, gen_context]
            )
        
        # Hybrid QA tab
        with gr.Tab("🔀 Hybrid QA"):
            gr.Markdown("Automatically choose the best approach")
            with gr.Row():
                with gr.Column():
                    hyb_question = gr.Textbox(
                        label="Question",
                        placeholder="What do you want to know?",
                        lines=2
                    )
                    hyb_context = gr.Textbox(
                        label="Context",
                        placeholder="Enter context text...",
                        lines=8
                    )
                    hyb_mode = gr.Radio(
                        ["Auto", "Extractive", "Generative"],
                        label="Mode",
                        value="Auto"
                    )
                    hyb_button = gr.Button("Answer", variant="primary")
                with gr.Column():
                    hyb_output = gr.Markdown(label="Answer")
            
            hyb_button.click(
                answer_hybrid,
                inputs=[hyb_question, hyb_context, hyb_mode],
                outputs=hyb_output
            )
            
            gr.Examples(
                examples=[
                    ["When was it built?", "The Eiffel Tower was constructed from 1887 to 1889. It has become a global cultural icon of France.", "Auto"],
                    ["What is AI?", "Artificial intelligence is intelligence demonstrated by machines, in contrast to natural intelligence displayed by humans.", "Auto"],
                ],
                inputs=[hyb_question, hyb_context, hyb_mode]
            )
    
    gr.Markdown("""
    ### 📖 Methods:
    - **Extractive:** Finds exact answer spans in context (fast, precise)
    - **Generative:** Generates answers (flexible, can rephrase)
    - **Hybrid:** Tries extractive first, falls back to generative
    """)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7862)
