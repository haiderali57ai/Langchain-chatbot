#with Tuple
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

llm = ChatOllama (
    model = "qwen3.5:cloud",
    temp = 0.7  
)
message =[
    
    ("system","You a helpful AI Assistant.")
    ("human","What is RA")
]
 
response = llm.invoke(message)
print(response.content)