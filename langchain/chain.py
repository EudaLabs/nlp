from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.chains import LLMChain

template = PromptTemplate(
    input_variables=["topic"],
    template="Please write a detailed explanation about {topic}."
)

llm = Ollama(
    model="llama3",  
    temperature=0.7
)

chain = LLMChain(llm=llm, prompt=template)

response = chain.invoke({"topic": "evolution of biology"})
print(response['text'])
