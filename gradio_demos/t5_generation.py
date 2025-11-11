"""
Gradio demo for T5 text generation.
"""

import gradio as gr
from t5_text_generation.model import T5TextGenerator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize model
model = T5TextGenerator(model_name="t5-small")


def summarize_text(text, max_length, min_length):
    """Summarize text."""
    try:
        summary = model.summarize(
            text,
            max_length=int(max_length),
            min_length=int(min_length)
        )
        return summary
    except Exception as e:
        return f"Error: {str(e)}"


def translate_text(text, source_lang, target_lang):
    """Translate text."""
    try:
        translation = model.translate(
            text,
            source_lang=source_lang,
            target_lang=target_lang
        )
        return translation
    except Exception as e:
        return f"Error: {str(e)}"


def paraphrase_text(text, num_variations):
    """Paraphrase text."""
    try:
        paraphrases = model.paraphrase(
            text,
            num_return_sequences=int(num_variations)
        )
        return "\n\n".join([f"{i+1}. {p}" for i, p in enumerate(paraphrases)])
    except Exception as e:
        return f"Error: {str(e)}"


def generate_question(context, answer):
    """Generate question from context and answer."""
    try:
        question = model.generate_question(context, answer if answer else None)
        return question
    except Exception as e:
        return f"Error: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="T5 Text Generation") as demo:
    gr.Markdown("# 🤖 T5 Text Generation")
    gr.Markdown("Comprehensive text-to-text generation using T5 models")
    
    with gr.Tabs():
        # Summarization tab
        with gr.Tab("📝 Summarization"):
            with gr.Row():
                with gr.Column():
                    sum_input = gr.Textbox(
                        label="Text to Summarize",
                        placeholder="Enter long text here...",
                        lines=10
                    )
                    sum_max_len = gr.Slider(20, 200, value=100, label="Max Length")
                    sum_min_len = gr.Slider(10, 100, value=30, label="Min Length")
                    sum_button = gr.Button("Summarize", variant="primary")
                with gr.Column():
                    sum_output = gr.Textbox(label="Summary", lines=10)
            
            sum_button.click(
                summarize_text,
                inputs=[sum_input, sum_max_len, sum_min_len],
                outputs=sum_output
            )
            
            gr.Examples(
                examples=[
                    ["Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of intelligent agents: any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals.", 100, 20],
                ],
                inputs=[sum_input, sum_max_len, sum_min_len]
            )
        
        # Translation tab
        with gr.Tab("🌍 Translation"):
            with gr.Row():
                with gr.Column():
                    trans_input = gr.Textbox(
                        label="Text to Translate",
                        placeholder="Enter text...",
                        lines=5
                    )
                    trans_source = gr.Textbox(label="Source Language", value="English")
                    trans_target = gr.Textbox(label="Target Language", value="German")
                    trans_button = gr.Button("Translate", variant="primary")
                with gr.Column():
                    trans_output = gr.Textbox(label="Translation", lines=5)
            
            trans_button.click(
                translate_text,
                inputs=[trans_input, trans_source, trans_target],
                outputs=trans_output
            )
            
            gr.Examples(
                examples=[
                    ["Hello, how are you today?", "English", "German"],
                    ["The weather is beautiful.", "English", "French"],
                ],
                inputs=[trans_input, trans_source, trans_target]
            )
        
        # Paraphrasing tab
        with gr.Tab("✍️ Paraphrasing"):
            with gr.Row():
                with gr.Column():
                    para_input = gr.Textbox(
                        label="Text to Paraphrase",
                        placeholder="Enter text...",
                        lines=5
                    )
                    para_num = gr.Slider(1, 5, value=3, step=1, label="Number of Variations")
                    para_button = gr.Button("Paraphrase", variant="primary")
                with gr.Column():
                    para_output = gr.Textbox(label="Paraphrases", lines=10)
            
            para_button.click(
                paraphrase_text,
                inputs=[para_input, para_num],
                outputs=para_output
            )
            
            gr.Examples(
                examples=[
                    ["Natural language processing is a fascinating field of study.", 3],
                    ["The quick brown fox jumps over the lazy dog.", 3],
                ],
                inputs=[para_input, para_num]
            )
        
        # Question Generation tab
        with gr.Tab("❓ Question Generation"):
            with gr.Row():
                with gr.Column():
                    qgen_context = gr.Textbox(
                        label="Context",
                        placeholder="Enter context...",
                        lines=5
                    )
                    qgen_answer = gr.Textbox(
                        label="Answer (optional)",
                        placeholder="Enter answer for specific question...",
                        lines=2
                    )
                    qgen_button = gr.Button("Generate Question", variant="primary")
                with gr.Column():
                    qgen_output = gr.Textbox(label="Generated Question", lines=3)
            
            qgen_button.click(
                generate_question,
                inputs=[qgen_context, qgen_answer],
                outputs=qgen_output
            )
            
            gr.Examples(
                examples=[
                    ["The Eiffel Tower is located in Paris, France. It was built in 1889.", "Paris"],
                    ["Python is a high-level programming language created by Guido van Rossum.", "Guido van Rossum"],
                ],
                inputs=[qgen_context, qgen_answer]
            )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
