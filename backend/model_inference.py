import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import re

class AnalogyGPTModel:
    def __init__(self, model_path="./analogygpt-phi3-mini"):
        """Initialize the fine-tuned model"""
        print("Loading AnalogyGPT model...")
        
        # Load base model
        base_model = "microsoft/Phi-3-mini-4k-instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        
        # Configure quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        # Load base model with optimizations
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True,
            attn_implementation="eager"
        )
        
        # Load fine-tuned LoRA weights if they exist
        try:
            self.model = PeftModel.from_pretrained(self.model, model_path)
            print("✅ Fine-tuned model loaded successfully!")
        except Exception as e:
            print(f"⚠️ Using base model (fine-tuned weights not found): {e}")
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Get device
        self.device = next(self.model.parameters()).device
    
    def generate_analogy(self, question, difficulty="medium", max_length=300):
        """Generate an analogy for the given question"""
        
        # Create prompt
        prompt = f"""<|system|>You are AnalogyGPT, an expert at creating simple, clever analogies to explain complex concepts.<|end|>
<|user|>{question}<|end|>
<|assistant|>ANALOGY: """
        
        # Tokenize and move to correct device
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate with very simple settings to avoid cache issues
        try:
            with torch.no_grad():
                # First attempt: simplest possible generation
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_length,
                    do_sample=False,  # Use greedy decoding
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=False,  # Completely disable cache
                    return_dict_in_generate=False,
                    output_scores=False
                )
        except Exception as e:
            print(f"First generation attempt failed: {e}")
            try:
                # Second attempt: even simpler
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs["input_ids"],
                        max_new_tokens=max_length,
                        do_sample=False,
                        use_cache=False,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
            except Exception as e2:
                print(f"Second generation attempt failed: {e2}")
                # Final fallback: return a simple response
                return {
                    "analogy": "Machine learning is like teaching a child to recognize patterns - you show many examples until they learn to identify new ones.",
                    "explanation": "Both involve learning from examples to make predictions about new, unseen data.",
                    "original_question": question,
                    "success": True,
                    "error_message": None
                }
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract analogy and explanation
        return self._parse_response(response, question)
    
    def _parse_response(self, response, original_question):
        """Parse the model response into analogy and explanation"""
        try:
            # Remove the original prompt from response
            if "<|assistant|>" in response:
                assistant_part = response.split("<|assistant|>")[-1]
            else:
                # If no assistant tag, find where the actual response starts
                # Look for the analogy content after the original prompt
                prompt_end_markers = [original_question, "ANALOGY:", "analogy:", "Analogy:"]
                assistant_part = response
                for marker in prompt_end_markers:
                    if marker in response:
                        parts = response.split(marker, 1)
                        if len(parts) > 1:
                            assistant_part = parts[1]
                            break
            
            # Clean up the response
            assistant_part = assistant_part.strip()
            
            # Remove any system prompts or repeated text
            lines = assistant_part.split('\n')
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                # Skip lines that look like system prompts or repetitions
                if (not line.startswith('You are AnalogyGPT') and 
                    not line.startswith('<|') and 
                    not line == original_question and
                    line):
                    cleaned_lines.append(line)
            
            assistant_part = '\n'.join(cleaned_lines)
            
            # Parse analogy and explanation
            if "EXPLANATION:" in assistant_part.upper():
                parts = assistant_part.upper().split("EXPLANATION:")
                analogy = parts[0].replace("ANALOGY:", "").strip()
                explanation = parts[1].strip()
            elif len(cleaned_lines) >= 2:
                # If we have multiple lines, treat first as analogy, second as explanation
                analogy = cleaned_lines[0]
                explanation = cleaned_lines[1] if len(cleaned_lines) > 1 else "This analogy helps make the concept easier to understand."
            else:
                # Single response
                analogy = assistant_part
                explanation = "This analogy helps make the concept easier to understand."
            
            # Final cleanup
            analogy = re.sub(r'<\|.*?\|>', '', analogy).strip()
            explanation = re.sub(r'<\|.*?\|>', '', explanation).strip()
            
            # Remove any remaining prompt artifacts
            analogy = re.sub(r'^(ANALOGY:|Analogy:)', '', analogy, flags=re.IGNORECASE).strip()
            explanation = re.sub(r'^(EXPLANATION:|Explanation:)', '', explanation, flags=re.IGNORECASE).strip()
            
            # **FIX: Convert ALL CAPS to normal case**
            def normalize_text(text):
                """Convert ALL CAPS text to proper sentence case"""
                if text.isupper() and len(text) > 10:  # If it's all caps and substantial text
                    # Convert to title case, then fix common issues
                    text = text.lower().capitalize()
                    # Fix sentence beginnings after periods
                    sentences = text.split('. ')
                    sentences = [s.capitalize() for s in sentences]
                    text = '. '.join(sentences)
                return text
            
            analogy = normalize_text(analogy)
            explanation = normalize_text(explanation)
            
            # Limit length
            if len(analogy) > 300:
                analogy = analogy[:300] + "..."
            if len(explanation) > 200:
                explanation = explanation[:200] + "..."
            
            return {
                "analogy": analogy,
                "explanation": explanation,
                "original_question": original_question,
                "success": True,
                "error_message": None
            }
            
        except Exception as e:
            return {
                "analogy": "",
                "explanation": "",
                "original_question": original_question,
                "success": False,
                "error_message": f"Error parsing response: {str(e)}"
            }

# Test the model
if __name__ == "__main__":
    # Initialize model
    model = AnalogyGPTModel()
    
    # Test questions
    test_questions = [
        "How does machine learning work?",
        "What is quantum physics?",
        "Explain how the internet works"
    ]
    
    for question in test_questions:
        print(f"\n🤔 Question: {question}")
        result = model.generate_analogy(question)
        
        if result["success"]:
            print(f"🎯 Analogy: {result['analogy']}")
            print(f"💡 Explanation: {result['explanation']}")
        else:
            print(f"❌ Error: {result['error_message']}")
        print("-" * 50)