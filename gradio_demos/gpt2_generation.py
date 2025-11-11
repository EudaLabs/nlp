"""
Gradio demo for GPT-2 text generation.
"""

import gradio as gr
from gpt2_text_generation.model import GPT2Generator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize model
model = GPT2Generator(model_name="gpt2")


def generate_text(prompt, max_length, temperature, top_p, repetition_penalty):
    """Generate text from prompt."""
    try:
        output = model.generate(
            prompt,
            max_length=int(max_length),
            temperature=float(temperature),
            top_p=float(top_p),
            repetition_penalty=float(repetition_penalty),
            num_return_sequences=1
        )[0]
        return output
    except Exception as e:
        return f"Error: {str(e)}"


def generate_story(prompt, length):
    """Generate a story."""
    try:
        story = model.generate_story(
            prompt,
            max_length=int(length),
            temperature=0.9
        )
        return story
    except Exception as e:
        return f"Error: {str(e)}"


def generate_variations(prompt, num_vars, temperature):
    """Generate variations."""
    try:
        variations = model.generate_variations(
            prompt,
            num_variations=int(num_vars),
            temperature=float(temperature),
            max_length=150
        )
        return "\n\n" + "\n\n".join([f"**Variation {i+1}:**\n{v}" for i, v in enumerate(variations)])
    except Exception as e:
        return f"Error: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="GPT-2 Text Generation") as demo:
    gr.Markdown("# ✨ GPT-2 Text Generation")
    gr.Markdown("Creative text generation using GPT-2")
    
    with gr.Tabs():
        # Basic Generation tab
        with gr.Tab("📝 Text Generation"):
            with gr.Row():
                with gr.Column():
                    gen_input = gr.Textbox(
                        label="Prompt",
                        placeholder="Start your text...",
                        lines=3
                    )
                    gen_length = gr.Slider(50, 500, value=150, label="Max Length")
                    gen_temp = gr.Slider(0.1, 2.0, value=0.8, step=0.1, label="Temperature")
                    gen_topp = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-p")
                    gen_rep = gr.Slider(1.0, 2.0, value=1.2, step=0.1, label="Repetition Penalty")
                    gen_button = gr.Button("Generate", variant="primary")
                with gr.Column():
                    gen_output = gr.Textbox(label="Generated Text", lines=15)
            
            gen_button.click(
                generate_text,
                inputs=[gen_input, gen_length, gen_temp, gen_topp, gen_rep],
                outputs=gen_output
            )
            
            gr.Examples(
                examples=[
                    ["Once upon a time in a distant galaxy,", 200, 0.9, 0.95, 1.2],
                    ["The future of artificial intelligence is", 150, 0.8, 0.9, 1.1],
                    ["In a world where technology had advanced beyond imagination,", 250, 1.0, 0.95, 1.2],
                ],
                inputs=[gen_input, gen_length, gen_temp, gen_topp, gen_rep]
            )
        
        # Story Generation tab
        with gr.Tab("📖 Story Generation"):
            with gr.Row():
                with gr.Column():
                    story_input = gr.Textbox(
                        label="Story Beginning",
                        placeholder="Start your story...",
                        lines=5
                    )
                    story_length = gr.Slider(100, 500, value=300, step=50, label="Story Length")
                    story_button = gr.Button("Generate Story", variant="primary")
                with gr.Column():
                    story_output = gr.Textbox(label="Generated Story", lines=20)
            
            story_button.click(
                generate_story,
                inputs=[story_input, story_length],
                outputs=story_output
            )
            
            gr.Examples(
                examples=[
                    ["In the year 2150, humanity discovered a signal from deep space.", 300],
                    ["The old lighthouse keeper had a secret that nobody knew.", 250],
                ],
                inputs=[story_input, story_length]
            )
        
        # Variations tab
        with gr.Tab("🔄 Variations"):
            with gr.Row():
                with gr.Column():
                    var_input = gr.Textbox(
                        label="Prompt",
                        placeholder="Enter prompt...",
                        lines=3
                    )
                    var_num = gr.Slider(1, 5, value=3, step=1, label="Number of Variations")
                    var_temp = gr.Slider(0.5, 2.0, value=1.2, step=0.1, label="Temperature (creativity)")
                    var_button = gr.Button("Generate Variations", variant="primary")
                with gr.Column():
                    var_output = gr.Markdown(label="Variations")
            
            var_button.click(
                generate_variations,
                inputs=[var_input, var_num, var_temp],
                outputs=var_output
            )
            
            gr.Examples(
                examples=[
                    ["The key to success is", 3, 1.2],
                    ["In the future, we will", 4, 1.3],
                ],
                inputs=[var_input, var_num, var_temp]
            )
    
    gr.Markdown("""
    ### 💡 Tips:
    - **Temperature**: Higher = more creative/random (try 0.7-1.5)
    - **Top-p**: Controls diversity (0.9-0.95 usually works well)
    - **Repetition Penalty**: Prevents repetition (1.1-1.3 recommended)
    """)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)
