# Earnings Call Compliment Extraction

This project extracts and validates compliments from earnings call transcripts using OpenAI's GPT models.

## Setup

### OpenAI API Key Setup

1. **Get your OpenAI API key**:
   - Go to https://platform.openai.com/api-keys
   - Create a new API key if you don't have one

2. **Create the API key file**:
   ```bash
   cp openai_api_key_template.txt openai_api_key.txt
   ```

3. **Add your API key**:
   - Open `openai_api_key.txt` in a text editor
   - Replace `YOUR_OPENAI_API_KEY_HERE` with your actual OpenAI API key
   - Save the file

4. **Security**: The `openai_api_key.txt` file is automatically excluded from git via `.gitignore` to keep your API key secure.

## Usage

### Extract Compliments for a Single Stock
```bash
python extractComplimentsFromEarningCalls.py
```

### Run Validation Only
```bash
python rerun_validation_only.py
```

## Files

- `extractComplimentsFromEarningCalls.py` - Main extraction script
- `activateChatty.py` - OpenAI API integration
- `rerun_validation_only.py` - Validation-only script
- `openai_api_key.txt` - Your API key (not committed to git)
- `openai_api_key_template.txt` - Template for setting up API key

## Configuration

Edit the `target_tickers` list in `extractComplimentsFromEarningCalls.py` to specify which stocks to process. 