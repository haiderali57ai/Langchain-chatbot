from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

llm = ChatOllama(
    model = "qwen3.5:cloud",
    temp = 0.7  
)
message =[
    SystemMessage(content="You a helpful AI Assistant."),
    #("system","You a helpful AI Assistant.")
    #("human","What is RA")
    HumanMessage(content="What is RAG" )
]
 
response = llm.invoke(message)
print(response.content)