from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(
    model = "qwen3.5:cloud",
    temp = 0.7  
)
prompt = ChatPromptTemplate.from_messagesmessage =[
    
    ("system","You a helpful AI Assistant.")
    ("human","{question}")
]
 
chain = prompt | llm | StrOutputParser() # Pipe operator is use for concatenation
#response = chain.invoke({"question":"What is RAG"})
#print(response)
for chunk in chain.stream({"question":"What is RAG"}):
    print(chunk, end="", flash=True)