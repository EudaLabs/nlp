from code_analyzer import analyze_code
from error_resolver import resolve_errors
from model import get_suggestions  # Function that analyzes using Llama3

def main():
    print("Welcome to the Code Debugger!")
    code_input = input("Please enter your Python code:\n")

    # Analyzing the code
    errors = analyze_code(code_input)

    if not errors:
        print("No syntax issues found. Running advanced analysis...")
        suggestions = get_suggestions(code_input)

    else:
        print("The following syntax issues were found in your code:")
        for error in errors:
            print(f"- {error}")

        suggestions = get_suggestions(code_input)
        print(f"Advanced Suggestions:\n{suggestions}")

        # Providing solutions for errors
        # print("\nSolutions:")
        # for error in errors:
        #     solution = resolve_errors(error)
        #     print(f"Solution for '{error}': {solution}")

if __name__ == "__main__":
    main()
