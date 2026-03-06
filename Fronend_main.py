from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

import os
from dotenv import load_dotenv

load_dotenv() #read .env  variables

MODEL_NAME = os.getenv("MODEL_NAME","qwen3.5:cloud")
TEMPERATURE = float(os.getenv("TEMPERATURE","0.7"))
Max_turns = int(os.getenv("MAX_TURNS", "10"))

llm = ChatOllama(
    model = "qwen3.5:cloud",
    temp = 0.7  
)
prompt = ChatPromptTemplate.from_messages ([
    
    ("system","You a helpful AI Assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human","{question}")
])
chain = prompt | llm | StrOutputParser()

chat_history =[] #Memory Store
#Max_turns = 10 # 10 messages (Human +AI)

def chat(question):
    current_turn = len(chat_history) // 2
    
    if current_turn >= Max_turns:
        return
        (
            "Context window is full"
            "AI may not follow previous thread properly"
            "Please type 'clear' for a new chat"
        )
    response = chain.invoke({
        "question":question,
        "chat_history":chat_history
    })
    
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=response))
    
    remaining = Max_turns - (current_turn + 1)
    if remaining <= 2:
        remaining += f"Warning: Only {remaining} turn(s) left to context i"

    return response

def main():
    print("LangChain Chatbot Ready! (Type 'quit' for exit, 'clear' for resetchat history)")
        
    while True:
            User_input = input("You: ").strip()
            
            if not User_input:
                continue
            if User_input.lower() == "quit":
                break
            if User_input.lower() == "clear":
                chat_history.clear()
                print("History cleared ,Starting fresh!")
                continue
            print(f"AI:{chat(User_input)}")
main()
            
#print(chat("What is RAG?"))
#print(chat("Give me a python example of it"))
#print(chat("Now explain the code you just gave"))

   
                
    
    