#!/usr/bin/env python3
"""
Setup script to prepare the environment for Gemma RAG application
"""
import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def check_ollama():
    """Check if Ollama is installed and running"""
    print("Checking Ollama installation...")
    try:
        result = subprocess.run("ollama --version", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Ollama is installed")
            return True
        else:
            print("❌ Ollama is not installed")
            return False
    except Exception as e:
        print(f"❌ Error checking Ollama: {e}")
        return False

def main():
    print("🚀 Setting up Gemma RAG Application")
    print("=" * 50)
    
    # Check if Ollama is installed
    if not check_ollama():
        print("\n📥 Please install Ollama first:")
        print("   Visit: https://ollama.ai/")
        print("   Or run: curl -fsSL https://ollama.ai/install.sh | sh")
        sys.exit(1)
    
    # Pull Gemma model
    print("\n📦 Pulling Gemma model...")
    gemma_success = run_command("ollama pull gemma:2b", "Pulling Gemma 2B model")
    
    if not gemma_success:
        print("⚠️  Failed to pull Gemma model. Please run manually:")
        print("   ollama pull gemma:2b")
    
    # Optionally pull embedding model
    print("\n📦 Pulling embedding model (optional)...")
    run_command("ollama pull nomic-embed-text", "Pulling embedding model")
    
    # Install Python dependencies
    print("\n📦 Installing Python dependencies...")
    pip_success = run_command("pip install -r requirements.txt", "Installing dependencies")
    
    if not pip_success:
        print("⚠️  Failed to install dependencies. Please run manually:")
        print("   pip install -r requirements.txt")
    
    # Create data directory
    if not os.path.exists("data"):
        os.makedirs("data")
        print("✅ Created data directory")
    
    print("\n🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Add your PDF files to the 'data' directory")
    print("2. Run: python populate_database.py")
    print("3. Run: streamlit run app.py")
    print("\nNote: The first query may take longer as models are loaded.")

if __name__ == "__main__":
    main()