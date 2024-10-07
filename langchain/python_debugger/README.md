İşte daha kısa bir README:

---

# Code Debugger with LangChain and Llama3

This is a Python code debugging tool using LangChain and the Llama3 model (via Ollama) for analyzing code and suggesting improvements.

## Features

- **Syntax Checking**: Detects syntax errors using Python's `ast` library.
- **Advanced Analysis**: Uses Llama3 to suggest code improvements beyond syntax.

## Project Structure

```
.
├── app.py               # Main application
├── code_analyzer.py     # Syntax error analysis
├── error_resolver.py    # Resolves errors
├── model.py             # LangChain and Llama3 integration
├── requirements.txt     # Dependencies
```

## Installation

1. Clone the repository and navigate into it:
   ```bash
   git clone https://github.com/yourusername/code-debugger-langchain.git
   cd code-debugger-langchain
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the application:
```bash
python app.py
```

- Enter your Python code when prompted.
- The tool first checks syntax; if no issues, it performs advanced analysis with Llama3.

## Example

**Input:**
```bash
print "Hello, World!"
```

**Output:**
```
Syntax issue: Missing parentheses in call to 'print'.
```

**Input:**
```bash
print("Hello, World!")
```

**Output:**
```
No syntax issues found. Advanced Suggestions: Code is correct.
```

## Troubleshooting

- **API Errors**: Ensure Ollama is set up properly.
- **Deprecation Warnings**: Update LangChain packages with:
  ```bash
  pip install -U langchain langchain-community
  ```

---

This version is concise and to the point. Let me know if further adjustments are needed!