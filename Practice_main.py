from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

# 1. Corrected 'temperature' and model name
llm = ChatOllama(
    model="qwen3.5:cloud", 
    temperature=0.7  
)

# 2. Corrected 'from_messages'
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI Assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

chain = prompt | llm | StrOutputParser()
chat_history = [] 
Max_turns = 10 

def chat(question):
    current_turn = len(chat_history) // 2
    
    # 3. Fixed logic: Check if turns EXCEED limit
    if current_turn >= Max_turns:
        return (
            "Context window is full. AI may not follow previous thread properly. "
            "Please type 'clear' for a new chat."
        )

    # 4. Fixed typo: "question"
    response = chain.invoke({
        "question": question,
        "chat_history": chat_history
    })
    
    # 5. Corrected order: Add Human first, then AI
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=response))
    
    return response

def main():
    print("LangChain Chatbot Ready! (Type 'quit' to exit, 'clear' to reset)")
        
    while True:
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "clear":
            chat_history.clear()
            print("History cleared, starting fresh!")
            continue
            
        print(f"AI: {chat(user_input)}")

if __name__ == "__main__":
    main()
