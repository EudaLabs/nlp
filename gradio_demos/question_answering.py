"""Gradio demo for question answering."""
import gradio as gr
import torch
from transformers import pipeline


class QuestionAnswerer:
    """Question answering model wrapper."""
    
    def __init__(self):
        """Initialize the QA model."""
        self.qa_pipeline = pipeline(
            "question-answering",
            model="distilbert-base-cased-distilled-squad",
            device=0 if torch.cuda.is_available() else -1,
        )
    
    def answer(self, question: str, context: str) -> tuple:
        """
        Answer a question based on context.
        
        Args:
            question: The question to answer
            context: The context containing the answer
        
        Returns:
            Tuple of (answer, confidence score)
        """
        if not question or not context:
            return "Please provide both question and context.", 0.0
        
        result = self.qa_pipeline(question=question, context=context)
        
        return result["answer"], result["score"]


# Initialize model
qa_model = QuestionAnswerer()


def answer_question(question: str, context: str) -> tuple:
    """Answer question based on context."""
    answer, confidence = qa_model.answer(question, context)
    
    # Format confidence as percentage
    confidence_text = f"Confidence: {confidence:.1%}"
    
    return answer, confidence_text


# Sample contexts
SAMPLE_CONTEXT_1 = """
The Amazon rainforest, also known as Amazonia, is a moist broadleaf tropical rainforest 
in the Amazon biome that covers most of the Amazon basin of South America. This basin 
encompasses 7,000,000 km2, of which 5,500,000 km2 are covered by the rainforest. 
The majority of the forest is contained within Brazil, with 60% of the rainforest, 
followed by Peru with 13%, and Colombia with 10%.
"""

SAMPLE_CONTEXT_2 = """
Artificial Intelligence (AI) is intelligence demonstrated by machines, as opposed to 
natural intelligence displayed by animals including humans. AI research has been defined 
as the field of study of intelligent agents, which refers to any system that perceives 
its environment and takes actions that maximize its chance of achieving its goals. 
The term "artificial intelligence" was coined in 1956 by John McCarthy.
"""

SAMPLE_CONTEXT_3 = """
The Apollo 11 mission was the first spaceflight that landed humans on the Moon. 
Commander Neil Armstrong and lunar module pilot Buzz Aldrin landed the Apollo Lunar 
Module Eagle on July 20, 1969, at 20:17 UTC. Armstrong became the first person to 
step onto the lunar surface six hours and 39 minutes later on July 21 at 02:56 UTC.
"""


# Create Gradio interface
demo = gr.Interface(
    fn=answer_question,
    inputs=[
        gr.Textbox(
            label="Question",
            placeholder="What do you want to know?",
            lines=2,
        ),
        gr.Textbox(
            label="Context",
            placeholder="Provide context that contains the answer...",
            lines=10,
        ),
    ],
    outputs=[
        gr.Textbox(label="Answer", lines=2),
        gr.Textbox(label="Confidence Score"),
    ],
    title="❓ Question Answering System",
    description="""
    Ask questions about any text and get answers extracted from the context.
    
    **How it works:**
    1. Provide a context (a paragraph or article)
    2. Ask a question about the context
    3. The model finds and extracts the answer
    
    **Note:** The answer must be present in the context. This is extractive QA, 
    not generative - it finds existing text rather than creating new answers.
    """,
    examples=[
        ["What percentage of the Amazon rainforest is in Brazil?", SAMPLE_CONTEXT_1],
        ["Which country has the second largest portion?", SAMPLE_CONTEXT_1],
        ["Who coined the term 'artificial intelligence'?", SAMPLE_CONTEXT_2],
        ["When was AI termed?", SAMPLE_CONTEXT_2],
        ["Who was the first person on the Moon?", SAMPLE_CONTEXT_3],
        ["When did the Apollo 11 landing occur?", SAMPLE_CONTEXT_3],
    ],
    theme=gr.themes.Soft(),
    allow_flagging="never",
)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=False,
    )
