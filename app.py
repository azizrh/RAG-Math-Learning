import streamlit as st
from query_data import query_rag
import os

# Set page configuration
st.set_page_config(
    page_title="RAG Chat Assistant",
    page_icon="🤖",
    layout="centered"
)

# Initialize session state for chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Custom CSS for better appearance
st.markdown("""
    <style>
    .stTextInput {
        padding: 20px 0px;
    }
    .stMarkdown {
        padding: 10px 0px;
    }
    .user-message {
        background-color: #e6f3ff;
        padding: 15px;
        border-radius: 10px;
        margin: 5px 0;
        color: black;
    }
    .assistant-message {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 5px 0;
        color: black;
    }
    .katex {
        font-size: 1.1em;
    }
    </style>
""", unsafe_allow_html=True)

# App title
st.title("📚 RAG Chat Assistant")
st.markdown("Ask questions about your documents and get AI-powered answers!")

# Display chat messages from history
for message in st.session_state.messages:
    with st.container():
        if message["role"] == "user":
            st.markdown(f"""
                <div class="user-message">
                    <b>You:</b><br>{message["content"]}
                </div>
            """, unsafe_allow_html=True)
        else:
            # Display assistant message with LaTeX support
            st.markdown('<div class="assistant-message"><b>Assistant:</b></div>', unsafe_allow_html=True)
            # Use st.markdown directly for proper LaTeX rendering
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get response from RAG system
    with st.spinner("Thinking..."):
        try:
            response = query_rag(prompt)
            
            # The response is now passed directly to st.markdown without any modification
            # This allows native rendering of LaTeX with single $ delimiters
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Force a rerun to update the chat display
            st.rerun()
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Add a clear chat button
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()
