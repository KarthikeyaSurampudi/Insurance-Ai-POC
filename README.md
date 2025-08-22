# Solidarity Insurance AI - Motor Claims Processing

A comprehensive AI-powered system for processing motor insurance claims, featuring document extraction, validation, and a dashboard interface.

## Features

- **Document Processing**: Extract key information from insurance documents (licenses, registrations, police reports, etc.)
- **AI-Powered Extraction**: Uses Azure OpenAI for intelligent document analysis
- **Validation Rules**: Cross-document validation for consistency checking
- **Dashboard Interface**: Streamlit-based web interface for claim management
- **Image Analysis**: Compare damage descriptions with actual vehicle photos

## Prerequisites

- Python 3.8+
- Azure OpenAI service access
- System dependencies:
  - `tesseract-ocr` for OCR functionality
  - `poppler-utils` for PDF processing

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Solidarity
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv insuranceAi
   source insuranceAi/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` file with your Azure OpenAI credentials:
   ```
   AZURE_ENDPOINT=your_azure_openai_endpoint_here
   AZURE_API_KEY=your_azure_openai_api_key_here
   AZURE_API_VER=2025-01-01-preview
   AZURE_MODEL=your_model_name_here
   ```

## Usage

### Running the Dashboard

Start the Streamlit dashboard:
```bash
streamlit run V03_Claims_Dashboard.py
```

The dashboard will be available at `http://localhost:8501`

### Processing Claims via Pipeline

Process claim documents using the pipeline:
```bash
python V03_Claims_Pipeline.py /path/to/claim/folder
```

### File Structure

- `V03_Claims_Dashboard.py` - Main dashboard application
- `V03_Claims_Pipeline.py` - Document processing pipeline
- `requirements.txt` - Python dependencies
- `.env` - Environment variables (not tracked in git)
- `.env.example` - Example environment variables

## Project Structure

```
Solidarity/
├── V03_Claims_Dashboard.py    # Streamlit dashboard
├── V03_Claims_Pipeline.py     # Document processing pipeline
├── requirements.txt           # Python dependencies
├── .env.example              # Environment variables template
├── claim_processed_records.csv # Master claim records
├── insurance_ai.log          # Application logs
└── [claim_id]_*.csv          # Per-claim output files
```

## Output Files

For each processed claim, the system generates:
- `[claim_id]_doc_checklist.csv` - Document presence checklist
- `[claim_id]_key_values.csv` - Extracted key-value pairs
- `[claim_id]_validations.csv` - Validation results
- `summary_[claim_id].txt` - Claim summary
- `accident_summary_[claim_id].txt` - Accident summary
- `[claim_id]_assets/` - Extracted vehicle damage photos

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `AZURE_ENDPOINT` | Azure OpenAI endpoint URL | `https://your-resource.openai.azure.com` |
| `AZURE_API_KEY` | Azure OpenAI API key | `your-api-key-here` |
| `AZURE_API_VER` | Azure API version | `2025-01-01-preview` |
| `AZURE_MODEL` | Deployment model name | `gpt-4` |

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is proprietary software developed for Solidarity Bahrain.

## Support

For technical support or questions, please contact the development team.
