from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from models import AnalogyRequest, AnalogyResponse, HealthCheck
from model_inference import AnalogyGPTModel
import uvicorn
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="AnalogyGPT API",
    description="A fine-tuned local AI model that creates clever analogies to explain complex concepts",
    version="2.0.0"
)

# Add CORS middleware to allow frontend to communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Global variables for model
analogy_model = None
model_loading = False
model_load_error = None

# Thread executor for model operations
executor = ThreadPoolExecutor(max_workers=1)

def load_model():
    """Load the model in a separate thread"""
    global analogy_model, model_loading, model_load_error
    
    try:
        print("🚀 Loading AnalogyGPT fine-tuned model...")
        model_loading = True
        analogy_model = AnalogyGPTModel()
        model_loading = False
        print("✅ AnalogyGPT model loaded successfully!")
        
    except Exception as e:
        model_loading = False
        model_load_error = str(e)
        print(f"❌ Error loading model: {e}")
        print("⚠️ Model failed to load - API will return errors")

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    # Start model loading in background thread
    threading.Thread(target=load_model, daemon=True).start()

@app.get("/", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    if model_loading:
        return HealthCheck(status="loading", message="AnalogyGPT model is loading... Please wait.")
    elif model_load_error:
        return HealthCheck(status="error", message=f"Model failed to load: {model_load_error}")
    elif analogy_model:
        return HealthCheck(status="ready", message="AnalogyGPT is ready with fine-tuned local model!")
    else:
        return HealthCheck(status="initializing", message="AnalogyGPT is initializing...")

@app.post("/generate-analogy", response_model=AnalogyResponse)
async def generate_analogy(request: AnalogyRequest):
    """Generate an analogy using the fine-tuned local model"""
    
    # Check model status
    if model_loading:
        return AnalogyResponse(
            analogy="",
            explanation="",
            original_question=request.question,
            success=False,
            error_message="Model is still loading. Please wait a moment and try again."
        )
    
    if model_load_error:
        return AnalogyResponse(
            analogy="",
            explanation="",
            original_question=request.question,
            success=False,
            error_message=f"Model failed to load: {model_load_error}"
        )
    
    if not analogy_model:
        return AnalogyResponse(
            analogy="",
            explanation="",
            original_question=request.question,
            success=False,
            error_message="Model not loaded yet. Please try again in a few moments."
        )
    
    try:
        # Generate analogy using local model in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor, 
            analogy_model.generate_analogy,
            request.question,
            request.difficulty_level or "medium"
        )
        
        return AnalogyResponse(
            analogy=result["analogy"],
            explanation=result["explanation"],
            original_question=result["original_question"],
            success=result["success"],
            error_message=result.get("error_message")
        )
        
    except Exception as e:
        print(f"Error during generation: {e}")
        return AnalogyResponse(
            analogy="",
            explanation="",
            original_question=request.question,
            success=False,
            error_message=f"Error generating analogy: {str(e)}"
        )

@app.get("/health")
async def simple_health():
    """Simple health check"""
    model_status = "loaded" if analogy_model else "loading" if model_loading else "error"
    return {
        "status": "ok", 
        "model": model_status,
        "model_error": model_load_error if model_load_error else None
    }

@app.get("/model-info")
async def model_info():
    """Get information about the loaded model"""
    return {
        "model_type": "Fine-tuned Phi-3-mini-4k-instruct",
        "training_examples": "2,457 analogies",
        "model_size": "3.8B parameters",
        "fine_tuning": "LoRA (Low-Rank Adaptation)",
        "training_loss": "0.55 (final)",
        "inference": "Local GPU (RTX 3050Ti)",
        "status": "loaded" if analogy_model else "loading" if model_loading else "error"
    }

@app.get("/test-analogy")
async def test_analogy():
    """Quick test endpoint"""
    if not analogy_model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = analogy_model.generate_analogy("How does a computer work?", "medium")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Test failed: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting AnalogyGPT Backend...")
    print("📦 Using fine-tuned Phi-3-mini model")
    print("🎯 Model will load in background...")
    print("🌐 Server starting on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)