import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "qwen3.5:cloud")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
MAX_TURNS = int(os.getenv("MAX_TURNS", "10"))

# Page Configuration
st.set_page_config(
    page_title="AI Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for stunning UI
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Main App Styling */
    .stApp {
        background: linear-gradient(135deg, #0d0d1a 0%, #1a1a2e 50%, #16213e 100%);
        font-family: 'Inter', sans-serif;
    }

    /* Title */
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 50%, #9b5de5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 10px;
        letter-spacing: -1px;
    }

    .subtitle {
        text-align: center;
        color: #8892b0;
        font-size: 1rem;
        margin-bottom: 30px;
    }

    /* Chat Messages */
    .chat-wrapper {
        max-width: 850px;
        margin: 0 auto;
    }

    .user-msg {
        background: linear-gradient(135deg, #3a7bd5 0%, #9b5de5 100%);
        color: white;
        padding: 18px 24px;
        border-radius: 24px 24px 8px 24px;
        margin: 12px 0;
        max-width: 75%;
        margin-left: auto;
        box-shadow: 0 8px 32px rgba(58, 123, 213, 0.35);
        border: 1px solid rgba(155, 93, 229, 0.3);
        font-size: 1rem;
        line-height: 1.5;
    }

    .ai-msg {
        background: linear-gradient(135deg, #232526 0%, #1a1a2e 100%);
        color: #e6e6e6;
        padding: 18px 24px;
        border-radius: 24px 24px 24px 8px;
        margin: 12px 0;
        max-width: 75%;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.08);
        font-size: 1rem;
        line-height: 1.5;
    }

    /* Message Avatar */
    .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
    }

    .user-avatar {
        background: linear-gradient(135deg, #3a7bd5 0%, #9b5de5 100%);
    }

    .ai-avatar {
        background: linear-gradient(135deg, #232526 0%, #414345 100%);
    }

    /* Input Area */
    .input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: rgba(13, 13, 26, 0.95);
        backdrop-filter: blur(20px);
        padding: 20px 30px;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        z-index: 100;
    }

    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.06);
        border: 2px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        color: white;
        padding: 16px 20px;
        font-size: 1rem;
        transition: all 0.3s ease;
    }

    .stTextInput > div > div > input:focus {
        border-color: #3a7bd5;
        box-shadow: 0 0 0 4px rgba(58, 123, 213, 0.2);
        outline: none;
    }

    .stTextInput > div > div > input::placeholder {
        color: rgba(255, 255, 255, 0.4);
    }

    /* Send Button */
    .stButton > button {
        background: linear-gradient(135deg, #3a7bd5 0%, #9b5de5 100%);
        color: white;
        border: none;
        border-radius: 14px;
        padding: 14px 28px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(58, 123, 213, 0.4);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(58, 123, 213, 0.5);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0d1a 0%, #1a1a2e 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    .sidebar-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: white;
        margin-bottom: 20px;
    }

    .sidebar-section {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 16px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Progress Bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #3a7bd5 0%, #9b5de5 100%);
    }

    /* Status Badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }

    .status-online {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        background: #10b981;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.6; transform: scale(1.1); }
    }

    /* Typing Indicator */
    .typing {
        display: flex;
        gap: 5px;
        padding: 15px 20px;
    }

    .typing span {
        width: 8px;
        height: 8px;
        background: #8892b0;
        border-radius: 50%;
        animation: typing 1.4s infinite;
    }

    .typing span:nth-child(2) { animation-delay: 0.2s; }
    .typing span:nth-child(3) { animation-delay: 0.4s; }

    @keyframes typing {
        0%, 60%, 100% { transform: translateY(0); }
        30% { transform: translateY(-10px); }
    }

    /* Welcome Card */
    .welcome-card {
        background: linear-gradient(135deg, rgba(58, 123, 213, 0.1) 0%, rgba(155, 93, 229, 0.1) 100%);
        border: 1px solid rgba(58, 123, 213, 0.2);
        border-radius: 24px;
        padding: 50px;
        text-align: center;
        max-width: 600px;
        margin: 50px auto;
    }

    .welcome-card h2 {
        color: white;
        margin-bottom: 15px;
    }

    .welcome-card p {
        color: #8892b0;
        font-size: 1.1rem;
    }

    /* Clear Button */
    .clear-btn {
        background: rgba(239, 68, 68, 0.15) !important;
        border: 1px solid rgba(239, 68, 68, 0.3) !important;
    }

    .clear-btn:hover {
        background: rgba(239, 68, 68, 0.25) !important;
    }

    /* Warning */
    .warning-box {
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 12px;
        padding: 12px 16px;
        color: #f59e0b;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize LLM
@st.cache_resource
def get_llm():
    return ChatOllama(
        model=MODEL_NAME,
        temperature=TEMPERATURE
    )

# Initialize chain
def get_chain():
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You a helpful AI Assistant."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    return prompt | llm | StrOutputParser()

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chain" not in st.session_state:
    st.session_state.chain = get_chain()

# Sidebar
with st.sidebar:
    st.markdown('<p class="sidebar-title">⚙️ Settings</p>', unsafe_allow_html=True)

    # Status
    st.markdown(f'''
    <div class="sidebar-section">
        <span class="status-badge status-online">
            <span class="status-dot"></span>
            Online
        </span>
    </div>
    ''', unsafe_allow_html=True)

    # Model Info
    st.markdown(f'''
    <div class="sidebar-section">
        <p style="color: #8892b0; font-size: 0.85rem; margin-bottom: 8px;">Model</p>
        <p style="color: white; font-weight: 600;">{MODEL_NAME}</p>
        <p style="color: #8892b0; font-size: 0.85rem; margin-top: 12px; margin-bottom: 8px;">Temperature</p>
        <p style="color: white; font-weight: 600;">{TEMPERATURE}</p>
    </div>
    ''', unsafe_allow_html=True)

    # Turn Counter
    current_turns = len(st.session_state.chat_history) // 2
    remaining = MAX_TURNS - current_turns

    st.markdown(f'''
    <div class="sidebar-section">
        <p style="color: #8892b0; font-size: 0.85rem; margin-bottom: 8px;">Conversation</p>
        <p style="color: white; font-weight: 600; font-size: 1.5rem;">{current_turns} <span style="color: #8892b0; font-size: 1rem;">/ {MAX_TURNS}</span></p>
    </div>
    ''', unsafe_allow_html=True)

    st.progress(min(current_turns / MAX_TURNS, 1.0))

    if remaining <= 2 and remaining > 0:
        st.markdown(f'<div class="warning-box">⚠️ Only {remaining} turns left!</div>', unsafe_allow_html=True)
    elif remaining == 0:
        st.markdown('<div class="warning-box">⚠️ Context full! Clear chat to continue.</div>', unsafe_allow_html=True)

    # Clear Button
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑️ Clear Chat", use_container_width=True, type="secondary"):
        st.session_state.chat_history = []
        st.rerun()

    # Tips
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('''
    <div class="sidebar-section">
        <p style="color: #8892b0; font-size: 0.85rem; margin-bottom: 10px;">💡 Tips</p>
        <ul style="color: #a0aec0; font-size: 0.9rem; padding-left: 20px; margin: 0;">
            <li>Ask follow-up questions</li>
            <li>Request code examples</li>
            <li>Discuss any topic</li>
        </ul>
    </div>
    ''', unsafe_allow_html=True)

# Main Content
st.markdown('<h1 class="main-title">🧠 AI Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Powered by LangChain + Ollama</p>', unsafe_allow_html=True)

# Chat Container
chat_container = st.container()

# Display Messages
with chat_container:
    for msg in st.session_state.chat_history:
        if isinstance(msg, HumanMessage):
            st.markdown(f'<div class="user-msg">{msg.content}</div>', unsafe_allow_html=True)
        elif isinstance(msg, AIMessage):
            st.markdown(f'<div class="ai-msg">{msg.content}</div>', unsafe_allow_html=True)

# Welcome State
if len(st.session_state.chat_history) == 0:
    st.markdown('''
    <div class="welcome-card">
        <h2>👋 Welcome!</h2>
        <p>Start a conversation with me. I'm here to help with questions, coding, explanations, and more!</p>
    </div>
    ''', unsafe_allow_html=True)

# Chat Input (fixed at bottom)
st.markdown("<br><br><br>", unsafe_allow_html=True)

col1, col2 = st.columns([6, 1])

with col1:
    user_input = st.text_input(
        "",
        placeholder="Type your message...",
        label_visibility="collapsed",
        key="input_field"
    )

with col2:
    send_btn = st.button("Send ➤", use_container_width=True)

# Handle submission
if send_btn and user_input.strip():
    current_turn = len(st.session_state.chat_history) // 2

    if current_turn >= MAX_TURNS:
        st.error("⚠️ Context window full! Please clear chat history.")
    else:
        # Add user message
        st.session_state.chat_history.append(HumanMessage(content=user_input))

        # Get response
        with st.spinner(""):
            try:
                response = st.session_state.chain.invoke({
                    "question": user_input,
                    "chat_history": st.session_state.chat_history[:-1]
                })
                st.session_state.chat_history.append(AIMessage(content=response))
            except Exception as e:
                st.error(f"Error: {str(e)}")

        st.rerun()

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown('''
<p style="text-align: center; color: #4a5568; font-size: 0.8rem;">
    Powered by LangChain + Ollama | {MODEL_NAME}
</p>
'''.format(MODEL_NAME=MODEL_NAME), unsafe_allow_html=True)
