import streamlit as st
from single_query_data import query_rag as single_query_rag
from multi_query_data import query_rag as multi_query_rag
from multi_query_multi_retrieval_stage import query_rag as multistage_query_rag
import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Advanced RAG Chat Assistant",
    page_icon="🤖",
    layout="centered"
)

# Initialize session state for chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize session state for query method if it doesn't exist
if "query_method" not in st.session_state:
    st.session_state.query_method = "Multi-Stage Retrieval"

# Initialize session state for API configuration
if "ollama_url" not in st.session_state:
    st.session_state.ollama_url = os.getenv("OLLAMA_API_URL", "http://localhost:11434")

if "selected_model" not in st.session_state:
    st.session_state.selected_model = os.getenv("OLLAMA_MODEL", "gemma3:4b")

if "available_models" not in st.session_state:
    st.session_state.available_models = []

if "stream_enabled" not in st.session_state:
    st.session_state.stream_enabled = False

# Function to fetch available models from Ollama
def fetch_available_models(api_url):
    """Fetch available models from Ollama API"""
    try:
        response = requests.get(f"{api_url}/api/tags", timeout=5)
        response.raise_for_status()
        models_data = response.json()
        models = [model["name"] for model in models_data.get("models", [])]
        return sorted(models) if models else ["gemma3:4b", "llama2:7b", "mistral:7b"]
    except Exception as e:
        st.warning(f"Could not fetch models from {api_url}: {str(e)}")
        return ["gemma3:4b", "llama2:7b", "mistral:7b", "codellama:7b"]

# Function to test API connection
def test_api_connection(api_url):
    """Test if the Ollama API is accessible"""
    try:
        response = requests.get(f"{api_url}/api/tags", timeout=3)
        return response.status_code == 200
    except:
        return False

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
    .method-info {
        background-color: #252d3d;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        border-left: 4px solid #ffc107;
    }
    .method-info-advanced {
        background-color: #1e3a2e;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        border-left: 4px solid #28a745;
    }
    .api-status {
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .api-connected {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        color: #155724;
    }
    .api-disconnected {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        color: #721c24;
    }
    .stSelectbox {
        margin-bottom: 20px;
    }
    .advanced-feature {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# App title
st.title("🚀 Advanced RAG Chat Assistant")
st.markdown("Ask questions about your documents with state-of-the-art retrieval methods!")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Configuration Section
    st.subheader("🌐 API Configuration")
    
    # Ollama API URL input
    new_api_url = st.text_input(
        "Ollama API URL:",
        value=st.session_state.ollama_url,
        help="Enter the URL of your Ollama server (e.g., http://localhost:11434)"
    )
    
    # Update API URL if changed
    if new_api_url != st.session_state.ollama_url:
        st.session_state.ollama_url = new_api_url
        st.session_state.available_models = []  # Reset models when URL changes
    
    # Test connection and fetch models
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh Models", use_container_width=True):
            with st.spinner("Fetching models..."):
                st.session_state.available_models = fetch_available_models(st.session_state.ollama_url)
    
    with col2:
        if st.button("🔍 Test Connection", use_container_width=True):
            with st.spinner("Testing connection..."):
                is_connected = test_api_connection(st.session_state.ollama_url)
                if is_connected:
                    st.success("✅ Connected!")
                    if not st.session_state.available_models:
                        st.session_state.available_models = fetch_available_models(st.session_state.ollama_url)
                else:
                    st.error("❌ Connection failed!")
    
    # Show connection status
    is_connected = test_api_connection(st.session_state.ollama_url)
    if is_connected:
        st.markdown("""
            <div class="api-status api-connected">
                <strong>🟢 API Status:</strong> Connected<br>
                <small>Server is responding normally</small>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="api-status api-disconnected">
                <strong>🔴 API Status:</strong> Disconnected<br>
                <small>Check your server URL and connection</small>
            </div>
        """, unsafe_allow_html=True)
    
    # Fetch models if not already loaded
    if not st.session_state.available_models:
        st.session_state.available_models = fetch_available_models(st.session_state.ollama_url)
    
    # Model selection
    st.subheader("🤖 Model Selection")
    selected_model = st.selectbox(
        "Choose Model:",
        options=st.session_state.available_models,
        index=st.session_state.available_models.index(st.session_state.selected_model) 
              if st.session_state.selected_model in st.session_state.available_models else 0,
        help="Select the language model to use for generating responses"
    )
    
    # Update selected model
    if selected_model != st.session_state.selected_model:
        st.session_state.selected_model = selected_model
    
    # Streaming option
    st.session_state.stream_enabled = st.checkbox(
        "🌊 Enable Streaming",
        value=st.session_state.stream_enabled,
        help="Enable real-time streaming of responses (slower but more interactive)"
    )
    
    st.markdown("---")
    
    # Query method selection with new option
    st.subheader("🔍 Query Method")
    query_method = st.selectbox(
        "Choose Query Method:",
        ["Single Query", "Multi-Query", "Multi-Stage Retrieval (Advanced)"],
        index=2 if st.session_state.query_method == "Multi-Stage Retrieval" else (1 if st.session_state.query_method == "Multi-Query" else 0),
        help="Choose your retrieval strategy based on your needs"
    )
    
    # Update session state if method changed
    if query_method != st.session_state.query_method:
        if query_method == "Multi-Stage Retrieval (Advanced)":
            st.session_state.query_method = "Multi-Stage Retrieval"
        else:
            st.session_state.query_method = query_method
    
    # Information about the selected method
    if st.session_state.query_method == "Single Query":
        st.markdown("""
            <div class="method-info">
                <strong>Single Query Mode:</strong><br>
                • Direct vector similarity search<br>
                • Fastest response time<br>
                • Good for specific, simple questions<br>
                • Best for known terminology
            </div>
        """, unsafe_allow_html=True)
    elif st.session_state.query_method == "Multi-Query":
        st.markdown("""
            <div class="method-info">
                <strong>Multi-Query Mode:</strong><br>
                • Generates multiple related queries<br>
                • Better document coverage<br>
                • More comprehensive results<br>
                • Good for complex questions
            </div>
        """, unsafe_allow_html=True)
    else:  # Multi-Stage Retrieval
        st.markdown("""
            <div class="method-info-advanced">
                <strong>🚀 Multi-Stage Retrieval:</strong><br>
                • AI-powered terminology discovery<br>
                • Automatic synonym detection<br>
                • Domain-adaptive query expansion<br>
                • Perfect for unknown datasets<br>
                • Handles complex domain knowledge
            </div>
        """, unsafe_allow_html=True)
        
        # Add advanced features highlight
        st.markdown("""
            <div class="advanced-feature">
                <strong>✨ Advanced Features Active:</strong><br>
                🧠 Terminology Analysis<br>
                🔍 Pattern Discovery<br>
                📈 Intelligent Expansion<br>
                🎯 Diversity Filtering
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Additional settings
    st.subheader("📊 Session Info")
    st.write(f"**Messages:** {len(st.session_state.messages)}")
    st.write(f"**Method:** {st.session_state.query_method}")
    st.write(f"**Model:** {st.session_state.selected_model}")
    st.write(f"**API:** {st.session_state.ollama_url}")
    st.write(f"**Streaming:** {'Yes' if st.session_state.stream_enabled else 'No'}")
    
    # Clear chat button in sidebar
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    # Export/Import Configuration
    st.subheader("💾 Configuration")
    
    config = {
        "ollama_url": st.session_state.ollama_url,
        "selected_model": st.session_state.selected_model,
        "query_method": st.session_state.query_method,
        "stream_enabled": st.session_state.stream_enabled
    }
    
    st.download_button(
        "📥 Export Config",
        data=json.dumps(config, indent=2),
        file_name="rag_config.json",
        mime="application/json",
        use_container_width=True
    )

# Main chat interface
st.markdown("---")

# Display current configuration info
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    if st.session_state.query_method == "Multi-Stage Retrieval":
        st.success("🚀 Multi-Stage: AI-powered retrieval active")
    elif st.session_state.query_method == "Multi-Query":
        st.info("🔍 Multi-Query: Questions expanded for better results")
    else:
        st.info("⚡ Single Query: Direct search for faster responses")

with col2:
    model_display = st.session_state.selected_model[:20] + "..." if len(st.session_state.selected_model) > 20 else st.session_state.selected_model
    st.info(f"🤖 Model: {model_display}")

with col3:
    methods = ["Single Query", "Multi-Query", "Multi-Stage Retrieval"]
    current_index = methods.index(st.session_state.query_method) if st.session_state.query_method in methods else 0
    if st.button("🔄 Next Method"):
        next_index = (current_index + 1) % len(methods)
        st.session_state.query_method = methods[next_index]
        st.rerun()

# Warning if API is not connected
if not is_connected:
    st.warning("⚠️ Warning: Cannot connect to Ollama API. Please check your configuration in the sidebar.")

# Special notice for Multi-Stage Retrieval
if st.session_state.query_method == "Multi-Stage Retrieval":
    st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; padding: 15px; border-radius: 10px; margin: 10px 0;">
            <strong>🧠 Multi-Stage Retrieval Active</strong><br>
            This advanced method analyzes your documents to discover terminology patterns and 
            automatically expands queries with relevant domain terms. Perfect for exploring 
            unfamiliar datasets or complex domains!
        </div>
    """, unsafe_allow_html=True)

# Display chat messages from history
for i, message in enumerate(st.session_state.messages):
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
            
            # Show which method and model was used for this response
            method_used = message.get("method", "Unknown")
            model_used = message.get("model", "Unknown")
            
            # Add special indicator for multi-stage retrieval
            if method_used == "Multi-Stage Retrieval":
                st.caption(f"🚀 Method: {method_used} | Model: {model_used} | Advanced AI Retrieval")
            else:
                st.caption(f"Method: {method_used} | Model: {model_used}")
            
            # Use st.markdown directly for proper LaTeX rendering
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know?", disabled=not is_connected):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Show current method being used with enhanced messaging
    method_display = st.session_state.query_method
    if st.session_state.query_method == "Multi-Stage Retrieval":
        spinner_text = f"🧠 Analyzing corpus with {st.session_state.selected_model}..."
    elif st.session_state.query_method == "Multi-Query":
        spinner_text = f"🔍 Generating multiple queries with {st.session_state.selected_model}..."
    else:
        spinner_text = f"⚡ Processing with {st.session_state.selected_model}..."
    
    with st.spinner(spinner_text):
        try:
            # Choose the appropriate query function based on selected method
            if st.session_state.query_method == "Multi-Stage Retrieval":
                # Show detailed progress for multi-stage
                progress_container = st.empty()
                progress_container.info("🚀 Initializing Multi-Stage Retrieval...")
                
                response = multistage_query_rag(
                    prompt, 
                    use_multiquery=False,  # Multi-stage handles its own query expansion
                    use_multistage=True,   # Enable multi-stage retrieval
                    ollama_url=st.session_state.ollama_url,
                    model=st.session_state.selected_model,
                    stream=st.session_state.stream_enabled
                )
                
                progress_container.empty()
                
            elif st.session_state.query_method == "Multi-Query":
                # Show additional info for multi-query
                status_placeholder = st.empty()
                status_placeholder.info("🔄 Generating multiple related queries...")
                
                response = multi_query_rag(
                    prompt, 
                    use_multiquery=True,
                    ollama_url=st.session_state.ollama_url,
                    model=st.session_state.selected_model,
                    stream=st.session_state.stream_enabled
                )
                
                status_placeholder.empty()
            else:
                response = single_query_rag(
                    prompt,
                    ollama_url=st.session_state.ollama_url,
                    model=st.session_state.selected_model,
                    stream=st.session_state.stream_enabled
                )
            
            # Add assistant response to chat history with method and model info
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "method": st.session_state.query_method,
                "model": st.session_state.selected_model
            })
            
            # Force a rerun to update the chat display
            st.rerun()
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Make sure your Ollama server is running and multi_query_multi_retrieval_stage.py is available")

# Footer with helpful information
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9em;">
        <p><strong>🚀 Advanced RAG Features:</strong></p>
        <p>• <strong>Multi-Stage Retrieval</strong>: AI discovers domain terminology automatically</p>
        <p>• <strong>Smart Expansion</strong>: Finds synonyms and related terms in your data</p>
        <p>• <strong>Adaptive Learning</strong>: Understands your corpus without manual configuration</p>
    </div>
""", unsafe_allow_html=True)

# Enhanced example queries section
with st.expander("💡 Example Queries & Method Recommendations"):
    tab1, tab2, tab3 = st.tabs(["🚀 Multi-Stage Examples", "🔍 Multi-Query Examples", "⚡ Single Query Examples"])
    
    with tab1:
        st.markdown("""
        **Perfect for Multi-Stage Retrieval:**
        - "What technical challenges are discussed?" *(discovers domain terminology)*
        - "How do different components interact?" *(finds relationships)*
        - "What methodologies are used?" *(identifies specialized terms)*
        - "What are the main performance metrics?" *(discovers domain-specific measures)*
        - "How is quality assessed?" *(finds evaluation terminology)*
        
        **Why Multi-Stage is powerful:**
        - 🧠 Analyzes your actual documents to understand terminology
        - 🔍 Discovers synonyms and related terms automatically  
        - 📈 Expands queries intelligently based on domain patterns
        - 🎯 Provides diverse, comprehensive results
        """)
    
    with tab2:
        st.markdown("""
        **Good for Multi-Query:**
        - "What are the main causes of climate change?"
        - "How does artificial intelligence impact society?"
        - "What are the benefits and risks of renewable energy?"
        
        **Multi-Query strengths:**
        - Generates multiple related queries
        - Better coverage of complex topics
        - Good when you know some domain terminology
        """)
    
    with tab3:
        st.markdown("""
        **Best for Single Query:**
        - "What is the capital of France?"
        - "Define machine learning"
        - "How do solar panels work?"
        
        **Single Query strengths:**
        - Fastest response time
        - Direct and precise
        - Good for simple factual questions
        """)

# Method comparison
with st.expander("📊 Method Comparison"):
    st.markdown("""
    | Feature | Single Query | Multi-Query | Multi-Stage Retrieval |
    |---------|-------------|-------------|----------------------|
    | **Speed** | ⚡⚡⚡ Fastest | ⚡⚡ Fast | ⚡ Moderate |
    | **Domain Adaptation** | ❌ None | ❌ Limited | ✅ Automatic |
    | **Synonym Handling** | ❌ Poor | ⚡ Basic | ✅ Advanced |
    | **Unknown Data** | ❌ Limited | ⚡ Moderate | ✅ Excellent |
    | **Result Quality** | ⚡ Good | ⚡⚡ Better | ✅✅ Best |
    | **Technical Complexity** | ⚡ Simple | ⚡⚡ Moderate | ⚡⚡⚡ Advanced |
    
    **Recommendation:** Use Multi-Stage Retrieval when exploring unfamiliar datasets or when you need the highest quality results!
    """)

# Debug information (optional - can be removed in production)
if st.sidebar.checkbox("🐛 Show Debug Info"):
    st.sidebar.subheader("Debug Information")
    st.sidebar.json({
        "Current Method": st.session_state.query_method,
        "Selected Model": st.session_state.selected_model,
        "API URL": st.session_state.ollama_url,
        "Streaming": st.session_state.stream_enabled,
        "Message Count": len(st.session_state.messages),
        "Available Models": st.session_state.available_models,
        "API Connected": is_connected
    })