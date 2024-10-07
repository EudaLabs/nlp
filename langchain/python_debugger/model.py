from langchain.prompts import PromptTemplate  # Correct module path
from langchain_community.llms import Ollama  # Importing Ollama from the community package

def create_langchain_model():
    # Creating the LangChain model and template
    template = """
    Analyze the following Python code and suggest any corrections or improvements(No more than 100 words):
    Code:
    {code}
    Suggestions:
    
    """

    llm = Ollama(model="llama3")  # Using the Llama3 model with Ollama
    prompt_template = PromptTemplate(template=template, input_variables=["code"])

    # Creating the chain using the prompt and model
    chain = prompt_template | llm

    return chain

def get_suggestions(code):
    chain = create_langchain_model()
    suggestions = chain.invoke({"code": code})  # Running the model with the correct argument format
    return suggestions