def resolve_errors(error):

    if "invalid syntax" in error:
        return "Check your syntax. Ensure all brackets, colons, and indentation are correct."
    elif "unexpected EOF while parsing" in error:
        return "It looks like your code is missing something. Make sure all code blocks are complete."
    else:
        return "No specific solution available. Please check your code for potential issues."


