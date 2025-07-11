# AnalogyGPT 🎯

A supervised fine-tuned AI application (on 2500+ training examples) that transforms complex concepts into simple, clever analogies. Built with a custom-trained Phi-3-mini model and modern web technologies.

## 🌟 Features

- **Fine-tuned Local AI Model**: Custom-trained Phi-3-mini with 2,457 analogy examples
- **Real-time Analogy Generation**: Instant explanations for complex topics
- **Modern Chat Interface**: Clean, ChatGPT-style UI with dark theme
- **No API Costs**: Completely local inference using your own hardware
- **Fast Performance**: Optimized for RTX 3050Ti with 4GB VRAM
- **Responsive Design**: Works seamlessly on desktop and mobile

## 🚀 Live Demo

frontend @ https://analogygpt-frontend.onrender.com/

Ask AnalogyGPT to explain concepts like:
- "How does machine learning work?"
- "What is quantum physics?"
- "Explain blockchain technology"
- "How does the human brain process memories?"

## 🛠️ Tech Stack

### Backend
- **FastAPI** - Modern Python web framework
- **Transformers** - Hugging Face library for model handling
- **PyTorch** - Deep learning framework
- **PEFT/LoRA** - Parameter-efficient fine-tuning
- **Phi-3-mini-4k-instruct** - Base model (3.8B parameters) which was tuned into a SFT Model

### Frontend
- **React** with TypeScript
- **Axios** - HTTP client
- **Lucide React** - Modern icons
- **CSS3** - Custom styling (no frameworks)

## 📊 Model Details

- **Base Model**: Microsoft Phi-3-mini-4k-instruct
- **Training Data**: 2,457 carefully curated analogy examples
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Training Loss**: 0.55 (final)
- **Parameters**: 3.8B total, 3.1M trainable (0.08%)
- **Hardware Requirements**: 4GB+ VRAM, 16GB+ RAM

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8+
- Node.js 16+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM
- 4GB+ VRAM

### Backend Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/analogygpt.git
cd analogygpt/backend
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate bitsandbytes peft trl pandas scikit-learn
pip install fastapi uvicorn python-dotenv
```

4. **Set up environment variables**
```bash
# Create .env file (optional - for OpenAI fallback)
echo "OPENAI_API_KEY=your_openai_key_here" > .env
```

5. **Train the model (optional - pre-trained weights included)**
```bash
python train_model.py  # Takes ~1 hour on RTX 3050Ti
```

6. **Start the backend**
```bash
python main.py
```

### Frontend Setup

1. **Navigate to frontend directory**
```bash
cd ../frontend
```

2. **Install dependencies**
```bash
npm install
```

3. **Start the development server**
```bash
npm start
```

4. **Open your browser**
```
http://localhost:3000
```

## 🐳 Docker Deployment

### Build and run with Docker Compose
```bash
docker-compose up --build
```

### Individual container builds
```bash
# Backend
docker build -t analogygpt-backend ./backend
docker run -p 8000:8000 analogygpt-backend

# Frontend
docker build -t analogygpt-frontend ./frontend
docker run -p 3000:3000 analogygpt-frontend
```

## ☁️ AWS Deployment

### Using EC2
1. Launch EC2 instance (g4dn.xlarge recommended for GPU)
2. Install Docker and dependencies
3. Clone repository and build containers
4. Configure security groups (ports 80, 443, 3000, 8000)
5. Set up reverse proxy with Nginx

### Using ECS
1. Build and push images to ECR
2. Create ECS cluster with GPU-enabled instances
3. Deploy services using provided task definitions
4. Configure Application Load Balancer

## 📁 Project Structure

```
analogygpt/
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── models.py               # Pydantic models
│   ├── train_model.py          # Model training script
│   ├── model_inference.py      # Inference logic
│   ├── training_data.csv       # Training dataset
│   ├── analogygpt-phi3-mini/   # Fine-tuned model weights
│   ├── requirements.txt        # Python dependencies
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatInterface.tsx
│   │   │   └── ChatInterface.css
│   │   ├── App.tsx
│   │   ├── App.css
│   │   └── index.css
│   ├── package.json
│   └── Dockerfile
├── docker-compose.yml
└── README.md
```

## 🎯 API Endpoints

### Health Check
```
GET /
GET /health
```

### Generate Analogy
```
POST /generate-analogy
Content-Type: application/json

{
  "question": "How does machine learning work?",
  "difficulty_level": "medium"
}
```

### Model Information
```
GET /model-info
```
## 📈 Performance Metrics

- **Inference Speed**: ~2-3 seconds per analogy
- **Memory Usage**: ~3.5GB VRAM, ~8GB RAM
- **Training Time**: ~90 minutes (2,457 examples)
- **Model Size**: ~2.4GB (quantized)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
