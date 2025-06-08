# Stock Chart Analysis with Local LLM

This application helps analyze stock charts for breakout patterns using a local LLM.

## Setup
### Option A: Using ollama:
1. Install Ollama from [https://ollama.ai/](https://ollama.ai/)
2. Pull the Llama3 model:
    ollama pull llama3
#### Option B: Using llama.cpp (more control over quantization)
1. Clone llama.cpp:
    git clone https://github.com/ggerganov/llama.cpp cd llama.cpp
2. Build it:
    mkdir build cd build cmake .. -DGGML_CUDA=ON cmake --build . --config Release

3. Download Llama 3 model files from [Meta's website](https://llama.meta.com/) or HuggingFace
4. Convert and quantize the model:
    python convert.py /path/to/llama/model --outtype f16 ./quantize /path/to/converted/model.bin q4_0 /path/to/output/model_q4_0.bin
3. Install Python dependencies:
    pip install -r requirements.txt
4. Install TA-Lib (for technical indicators):
- Windows: Download from [https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib)
- Mac: `brew install ta-lib`
- Linux: `apt-get install ta-lib`

## Running the Application

1. Start Ollama:
    - If using Ollama: `ollama serve`
    - If using llama.cpp: Run the server with your model
2. Launch the Streamlit UI:
    cd ui streamlit run streamlit_app.py
3. Open your browser to http://localhost:8501

## Features

- Scan stocks for breakout patterns based on momentum trend following criteria
- Generate entry, stop loss, and take profit levels
- Position sizing based on risk management rules
- LLM-enhanced chart analysis
- Watchlist management