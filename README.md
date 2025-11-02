# Engineering Items Attributes Enrichment

AI-powered system for extracting structured data from engineering PDF drawings using computer vision, OCR, and optional LLM enhancement.

## Features

- **PDF Processing**: Extract pages and embedded text
- **OCR Pipeline**: Dual-engine OCR (PaddleOCR + EasyOCR) with fallback
- **Entity Extraction**: Extract part numbers, materials, dimensions, etc.
- **Shape Detection**: YOLOv8-based component detection
- **LLM Enhancement**: Optional GPT-4/Claude integration with budget controls
- **Smart Routing**: Automatic pipeline selection based on confidence scores
- **Cost Optimization**: Automatic model step-down when budget threshold reached

## Setup

1. **Clone and navigate to project**
``````bash
cd engineering_items_attributes_enrichment
``````

2. **Activate conda environment**
``````bash
conda activate ai_project
``````

3. **Install dependencies**
``````bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
``````

4. **Configure API keys**
``````bash
cp .env.example .env
# Edit .env and add your API keys
``````

5. **Initialize database**
``````bash
python -m drawing_intelligence.database.manager init
``````

## Usage

### Process Single Drawing
``````bash
python main.py process --input drawing.pdf --output results/
``````

### Process Batch
``````bash
python main.py batch --input-dir drawings/ --output-dir results/ --workers 4
``````

### Export Results
``````bash
python main.py export --drawing-ids DWG-001,DWG-002 --format json --output export.json
``````

## Project Structure
``````
src/drawing_intelligence/
├── models/              # Model registry and specifications
├── llm/                 # LLM integration and budget control
├── orchestration/       # Pipeline orchestration and routing
├── processing/          # Core processing modules (OCR, entity extraction, etc.)
├── database/            # Database management
├── export/              # Export utilities
├── quality/             # Quality scoring and validation
└── utils/               # Utilities
``````

## Development

Run tests:
``````bash
pytest tests/ -v
``````

Format code:
``````bash
black src/ tests/
``````

## License

[Your License]