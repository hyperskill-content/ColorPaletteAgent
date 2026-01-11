# Color Palette Generator

An AI-powered color palette extraction tool that uses multiple approaches—from traditional K-Means clustering to advanced multi-model spatial reasoning with GPT-4o and Qwen-VL.

## Features

- 🎨 **Multiple Agents**: From classic ML to advanced AI-powered extraction
- 📐 **Smart Resizing**: Automatically resizes images to 1024px on the larger side
- 🖼️ **Visual Output**: Side-by-side visualizations with numbered sampling points
- 📊 **JSON Metadata**: Complete color data with RGB, hex, percentages, and coordinates
- 🏗️ **Modular Architecture**: Abstract base class with factory pattern for easy extensibility
- 🏷️ **Experiment Tracking**: Tag experiments and organize outputs by agent version
- 🌐 **Web Interface**: Minimalistic Next.js frontend with drag-and-drop upload

## Installation

This project uses `uv` for Python package management:

```bash
# Create virtual environment
uv venv

# Install dependencies
uv pip install -r requirements.txt
```

For the web app frontend:

```bash
cd frontend
npm install
```

## Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
```

## Available Agents

| Agent | Version | Description |
|-------|---------|-------------|
| `classic-ml` | V1 | K-Means clustering on all pixels |
| `simple-ai` | V2 | GPT-4o vision generates colors directly (trusts AI hex) |
| `advanced-v3` | V3 | GPT-4o describes + Qwen-VL localizes + pixel sampling |
| `exploratory-v4` | V4 | GPT-4o desc + Qwen-VL point + 60x60 KMeans + GPT-4o vibe pick |

## CLI Usage

### Basic Commands

```bash
# Process a single image with default agent (classic-ml)
uv run python main.py --image images/red-car.jpg

# Process all images in the images/ folder
uv run python main.py --all

# Add experiment tag for organization
uv run python main.py --all --tag "my-experiment"

# Customize number of colors
uv run python main.py --all --colors 7
```

### Testing Different Agents

```bash
# Classic ML (K-Means clustering)
uv run python main.py --agent classic-ml --all --tag "kmeans-test"

# Simple AI (GPT-4o only, trusts AI hex output)
uv run python main.py --agent simple-ai --all --tag "gpt4o-test"

# Advanced V3 (GPT-4o + Qwen-VL spatial sampling)
uv run python main.py --agent advanced-v3 --all --tag "v3-spatial"

# Exploratory V4 (Full pipeline with vibe matching)
uv run python main.py --agent exploratory-v4 --all --tag "v4-vibe"
```

### Full Options

```bash
uv run python main.py --help
```

## Running the Web App

The web app consists of a FastAPI backend and a Next.js frontend.

### 1. Start the Backend

```bash
cd backend
uv run uvicorn api:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

### 2. Start the Frontend

```bash
cd frontend
npm run dev
```

The web interface will be available at `http://localhost:3000`.

### Web App Features

- **Drag & Drop**: Upload images directly from your computer
- **URL Input**: Paste any image URL to process remote images
- **Interactive Palette**: Click any color card to copy the hex code
- **Spatial Visualization**: Numbered markers show where each color was sampled
- **V4 Intelligence**: Uses the most advanced `exploratory-v4` agent by default

## Project Structure

```
ColorPalette-Hyperskill/
├── backend/
│   ├── agent.py          # All color palette agents
│   └── api.py            # FastAPI endpoints
├── frontend/
│   ├── app/
│   │   ├── page.tsx      # Main interactive UI
│   │   ├── layout.tsx    # Root layout
│   │   └── globals.css   # Tailwind styles
│   ├── package.json
│   └── tailwind.config.ts
├── images/               # Sample input images
├── output/               # Generated palettes (auto-created)
├── main.py               # CLI entry point
├── requirements.txt      # Python dependencies
└── .env                  # API keys (create this)
```

## Agent Architecture

### V1: ClassicMLColorPaletteAgent
- Uses scikit-learn's K-Means clustering
- Extracts dominant colors mathematically
- Fast and deterministic

### V2: SimpleColorPaletteAgent
- Single GPT-4o call with image
- AI generates hex codes directly
- Quick but may "hallucinate" colors

### V3: AdvancedColorPaletteAgent
- GPT-4o generates vivid descriptions
- Qwen-VL localizes exact pixel coordinates
- Colors sampled from actual pixels (no hallucinations)

### V4: ExploratoryColorPaletteAgent
- GPT-4o generates vivid descriptions
- Qwen-VL gives rough starting coordinates
- 60x60 crop around point, KMeans extracts 5 distinct colors
- GPT-4o picks the color that best matches the "vibe"
- Most robust and semantically accurate

## Output Format

Each run generates:

1. **PNG Visualization**: Side-by-side view with numbered points
2. **JSON Metadata**:
   - RGB values
   - Hex codes
   - Percentages
   - Pixel coordinates
   - Timestamps
   - Agent information

## Requirements

### Python
- Python 3.13+
- numpy, matplotlib, pillow, scikit-learn
- fastapi, uvicorn (for web app)
- openai, python-dotenv (for AI agents)

### Node.js (for frontend)
- Node.js 18+
- Next.js 16+
- React, Tailwind CSS

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/process-upload` | POST | Upload an image file |
| `/process-url` | POST | Process image from URL |

### Example API Call

```bash
curl -X POST "http://localhost:8000/process-url" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/image.jpg"}'
```

## License

MIT
