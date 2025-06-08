from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os

class ChartAnalysisLLM:
    def __init__(self, local_model=True, server_url=None):
        if local_model:
            # Direct model loading (more control but requires more setup)
            model_path = "C:\\Users\\mmcar\\Desktop\\Dev\\ChartAnalysis\\llama.cpp\\models\\Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"

            
            callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
            
            self.llm = LlamaCpp(
                model_path=model_path,
                temperature=0.1,
                max_tokens=2048,
                n_ctx=4096,
                top_p=0.95,
                callback_manager=callback_manager,
                n_gpu_layers=-1,  # Use all available GPU layers
                verbose=False,
            )
        else:
            # Using server (recommended for stability)
            from langchain.llms import OpenAI
            
            if not server_url:
                server_url = "http://localhost:8000/v1"
                
            self.llm = OpenAI(
                openai_api_key="sk-no-key-needed",
                openai_api_base=server_url,
                max_tokens=2048,
                temperature=0.1,
            )
    
    def analyze_chart(self, chart_data, sector_info):
        """Analyze chart data for breakout patterns"""
        prompt = f"""
        You are a professional stock trader specializing in breakout patterns and technical analysis.
        
        Analyze this stock chart data and determine if it matches criteria for a Momentum Trend Following Breakout:
        
        Chart Data: {chart_data}
        Sector Info: {sector_info}
        
        Criteria to check:
        1. Prior move of 30%+ before consolidation
        2. 3-20+ days of consolidation
        3. Series of narrow range days before breakout
        4. Stock above/surfing moving averages (10, 20, 50, 200)
        5. Volume patterns (higher on breakout days, diminishing volume on consolidation)
        6. Closes at or near HOD on breakout days
        7. Not extended after recent consolidation
        8. Linear and orderly moves (not barcode pattern)
        
        Provide: 
        - Breakout potential (High/Medium/Low)
        - Entry price target
        - Stop loss recommendation (low of day)
        - Position sizing based on 2% max risk
        - Key levels to watch
        """
        
        return self.llm.invoke(prompt)