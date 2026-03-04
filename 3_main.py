from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOllama(
    model = "qwen3.5:cloud",
    temp = 0.7  
)
prompt = ChatPromptTemplate.from_messagesmessage =[
    
    ("system","You a helpful AI Assistant.")
    ("human","{question}")
]
 
chain = prompt | llm
response = chain.invoke({"question":"What is RAG"})
print(response.content)