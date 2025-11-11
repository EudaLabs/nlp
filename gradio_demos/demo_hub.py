"""
NLP Demo Hub - Launch all Gradio demos from one place.
"""

import gradio as gr
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from t5_text_generation.model import T5TextGenerator
from gpt2_text_generation.model import GPT2Generator
from question_answering.model import ExtractiveQA, GenerativeQA
from advanced_text_classification.model import ZeroShotClassifier
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize models (lightweight versions for demo hub)
logger.info("Loading models...")
t5_model = T5TextGenerator(model_name="t5-small")
gpt2_model = GPT2Generator(model_name="gpt2")
qa_model = ExtractiveQA()
zero_shot = ZeroShotClassifier()
logger.info("Models loaded!")


# T5 Functions
def t5_summarize(text):
    return t5_model.summarize(text, max_length=100)


def t5_translate(text):
    return t5_model.translate(text, "English", "German")


# GPT-2 Functions
def gpt2_generate(prompt):
    return gpt2_model.generate(prompt, max_length=150, temperature=0.8)[0]


# QA Functions
def qa_answer(question, context):
    answers = qa_model.answer(question, context, top_k=1)
    return f"{answers[0]['answer']} (confidence: {answers[0]['score']:.4f})"


# Zero-shot Functions
def zero_shot_classify(text, labels):
    label_list = [l.strip() for l in labels.split(',')]
    result = zero_shot.classify(text, label_list)
    return f"{result['labels'][0]} ({result['scores'][0]:.4f})"


# Create unified demo
with gr.Blocks(title="NLP Demo Hub", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🚀 NLP Demo Hub
    ### Comprehensive Natural Language Processing Demonstrations
    
    Explore various NLP capabilities in one place!
    """)
    
    with gr.Tabs():
        # T5 Generation
        with gr.Tab("📝 T5 - Summarization"):
            gr.Markdown("### Text Summarization with T5")
            with gr.Row():
                with gr.Column():
                    t5_sum_input = gr.Textbox(
                        label="Text to Summarize",
                        placeholder="Enter long text...",
                        lines=8
                    )
                    t5_sum_btn = gr.Button("Summarize", variant="primary")
                with gr.Column():
                    t5_sum_output = gr.Textbox(label="Summary", lines=8)
            
            t5_sum_btn.click(t5_summarize, inputs=t5_sum_input, outputs=t5_sum_output)
            
            gr.Examples(
                examples=[[
                    "Artificial intelligence (AI) is intelligence demonstrated by machines, "
                    "in contrast to natural intelligence displayed by humans and animals. "
                    "Leading AI textbooks define the field as the study of intelligent agents."
                ]],
                inputs=t5_sum_input
            )
        
        with gr.Tab("🌍 T5 - Translation"):
            gr.Markdown("### English to German Translation")
            with gr.Row():
                with gr.Column():
                    t5_trans_input = gr.Textbox(
                        label="English Text",
                        placeholder="Enter English text...",
                        lines=5
                    )
                    t5_trans_btn = gr.Button("Translate", variant="primary")
                with gr.Column():
                    t5_trans_output = gr.Textbox(label="German Translation", lines=5)
            
            t5_trans_btn.click(t5_translate, inputs=t5_trans_input, outputs=t5_trans_output)
            
            gr.Examples(
                examples=[["Hello, how are you today?"], ["The weather is beautiful."]],
                inputs=t5_trans_input
            )
        
        # GPT-2 Generation
        with gr.Tab("✨ GPT-2 - Text Generation"):
            gr.Markdown("### Creative Text Generation with GPT-2")
            with gr.Row():
                with gr.Column():
                    gpt2_input = gr.Textbox(
                        label="Prompt",
                        placeholder="Start your text...",
                        lines=3
                    )
                    gpt2_btn = gr.Button("Generate", variant="primary")
                with gr.Column():
                    gpt2_output = gr.Textbox(label="Generated Text", lines=10)
            
            gpt2_btn.click(gpt2_generate, inputs=gpt2_input, outputs=gpt2_output)
            
            gr.Examples(
                examples=[
                    ["Once upon a time in a distant galaxy,"],
                    ["The future of artificial intelligence is"],
                    ["In a world where technology had advanced beyond imagination,"]
                ],
                inputs=gpt2_input
            )
        
        # Question Answering
        with gr.Tab("❓ Question Answering"):
            gr.Markdown("### Answer Questions from Context")
            with gr.Row():
                with gr.Column():
                    qa_context = gr.Textbox(
                        label="Context",
                        placeholder="Enter context...",
                        lines=6
                    )
                    qa_question = gr.Textbox(
                        label="Question",
                        placeholder="What do you want to know?",
                        lines=2
                    )
                    qa_btn = gr.Button("Answer", variant="primary")
                with gr.Column():
                    qa_output = gr.Textbox(label="Answer", lines=3)
            
            qa_btn.click(qa_answer, inputs=[qa_question, qa_context], outputs=qa_output)
            
            gr.Examples(
                examples=[[
                    "The Eiffel Tower is located in Paris, France. It was built in 1889.",
                    "Where is the Eiffel Tower?"
                ]],
                inputs=[qa_context, qa_question]
            )
        
        # Zero-shot Classification
        with gr.Tab("🎯 Zero-Shot Classification"):
            gr.Markdown("### Classify Without Training")
            with gr.Row():
                with gr.Column():
                    zs_text = gr.Textbox(
                        label="Text to Classify",
                        placeholder="Enter text...",
                        lines=5
                    )
                    zs_labels = gr.Textbox(
                        label="Candidate Labels (comma-separated)",
                        placeholder="positive, negative, neutral",
                        value="positive, negative, neutral"
                    )
                    zs_btn = gr.Button("Classify", variant="primary")
                with gr.Column():
                    zs_output = gr.Textbox(label="Classification", lines=3)
            
            zs_btn.click(zero_shot_classify, inputs=[zs_text, zs_labels], outputs=zs_output)
            
            gr.Examples(
                examples=[
                    ["This movie was absolutely fantastic! I loved every minute of it.", "positive, negative, neutral"],
                    ["The product broke after one week. Very disappointed.", "positive, negative, neutral"]
                ],
                inputs=[zs_text, zs_labels]
            )
    
    gr.Markdown("""
    ---
    ### 🔗 Individual Demos
    
    For more detailed features, launch individual demos:
    - **T5 Generation**: `python -m gradio_demos.t5_generation`
    - **GPT-2 Generation**: `python -m gradio_demos.gpt2_generation`
    - **Question Answering**: `python -m gradio_demos.question_answering_demo`
    - **Sentiment Analysis**: `python -m gradio_demos.sentiment_analysis`
    
    ### 📚 Documentation
    
    Visit the README files in each project directory for detailed documentation.
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
