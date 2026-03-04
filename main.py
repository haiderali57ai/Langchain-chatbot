from langchain_ollama import ChatOllama

llm = ChatOllama(
    model = "qwen3.5:cloud",
    temp = 0.7  
)
 
response = llm.invoke("What is Al")
print(response.content)