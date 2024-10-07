import gradio as gr
from code_analyzer import analyze_code
from error_resolver import resolve_errors
from model import get_suggestions  # Function that analyzes using Llama3

def analyze_code_input(code_input):
    # Analyzing the code
    errors = analyze_code(code_input)

    if not errors:
        suggestions = get_suggestions(code_input)
        return "No syntax issues found.", suggestions, ""
    else:
        error_messages = "\n".join(errors)
        solutions = "\n".join([resolve_errors(error) for error in errors])
        suggestions = get_suggestions(code_input)
        return f"The following syntax issues were found:\n{error_messages}", suggestions, solutions

# Defining the Gradio interface
iface = gr.Interface(
    fn=analyze_code_input,  # Function
    inputs="text",  # Receiving code input as text from the user
    outputs=["text", "text", "text"],  # Outputs: Error message, suggestions, solutions
    title="Python Code Debugger",
    description="Enter your Python code below to analyze it for syntax issues and receive suggestions."
)

# Launching the Gradio application
if __name__ == "__main__":
    iface.launch()
