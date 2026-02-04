import requests

class LLMHandler:
    def __init__(self):
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model = "llama2"
    
    def generate_response(self, query: str, context: str = ""):
        """Generate response using Ollama"""
        prompt = f"""You are Jarvis, a helpful AI assistant.

Context: {context}

User: {query}

Jarvis:"""
        
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()['response']
            else:
                return "I'm having trouble connecting to my AI model. Please make sure Ollama is running."
                
        except Exception as e:
            return f"Error: {str(e)}. Make sure Ollama is installed and running (ollama serve)."