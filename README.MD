# Offline Generative AI Chatbot with Conversation History

A lightweight, terminal-based Generative AI chatbot that runs a locally saved language model on your computer. It uses **Python**, **PyTorch**, and **Hugging Face Transformers** to generate responses without calling an external inference API.

> **Model:** `HuggingFaceTB/SmolLM2-360M-Instruct`  
> **Execution:** CPU  
> **Interface:** Command-line / Jupyter Notebook

## Features

- **Local text generation:** Loads the model from the `smollm2_model` directory.
- **Offline chat after setup:** Once the model files have been downloaded and saved locally, chat generation can run without an internet connection.
- **Conversation context:** Sends recent user and assistant messages to the model to help it respond in context.
- **Reset command:** Clear the current conversation using `/reset`.
- **Exit command:** End the chat using `/exit`.
- **Bounded history:** Keeps the conversation history limited to recent messages.
- **Teaching-assistant prompt:** Encourages simple English explanations and helpful examples.

## How It Works

1. **Download the model:** The notebook downloads the tokenizer and model files from Hugging Face.
2. **Save the model locally:** The files are saved in the `smollm2_model/` folder.
3. **Load the local model:** The chatbot loads the tokenizer and model with `local_files_only=True`.
4. **Prepare the prompt:** A system prompt, recent conversation messages, and the new question are formatted using the model's chat template.
5. **Generate a response:** PyTorch runs text generation on the CPU.
6. **Continue the conversation:** The new question and answer are added to the in-memory conversation history.

## Project Structure

```text
Offline-GenAI-Chatbot/
├── Offline Generative AI Chatbot with Conversation History.ipynb
└── smollm2_model/
    ├── config.json
    ├── tokenizer files
    └── model weight files
```

The `smollm2_model/` directory is created when you run the model-download cells. Exact files may vary by Transformers/model version. The model directory can be large, so it is usually better not to commit it to GitHub unless you specifically intend to distribute the model files.

## Requirements

- Python 3.10 or a compatible Python version supported by the installed PyTorch and Transformers releases
- Jupyter Notebook or JupyterLab
- PyTorch
- Hugging Face Transformers
- Sufficient RAM and disk space for the model

The notebook uses CPU-only PyTorch. Model generation speed depends on your CPU and available memory.

## Installation and Setup

### 1. Clone or download the project

If you have uploaded the notebook to a Git repository:

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

Alternatively, open the notebook file directly in Jupyter.

### 2. Install the dependencies

Run the following commands in a notebook cell:

```python
%pip install --upgrade pip
%pip install torch --index-url https://download.pytorch.org/whl/cpu
%pip install --upgrade transformers
```

### 3. Download and save the model

Run the model-download cell in the notebook while connected to the internet:

```python
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
MODEL_DIR = Path("smollm2_model")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

tokenizer.save_pretrained(MODEL_DIR)
model.save_pretrained(MODEL_DIR, safe_serialization=True)

print("Model saved to:", MODEL_DIR.resolve())
```

This initial download requires an internet connection. Keep the `smollm2_model/` folder available on the computer where you want to run the chatbot.

### 4. Load the model locally

Run the notebook cell that loads the local model:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.set_num_threads(4)

MODEL_DIR = "smollm2_model"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)

model.to("cpu")
model.eval()
```

### 5. Start chatting

Run all notebook cells in order, then execute the cell containing:

```python
start_chat()
```

## Chat Commands

| Command | Description |
|---|---|
| `/reset` | Clears the current conversation history |
| `/exit` | Ends the chatbot session |
| Empty input | Ignored; the chatbot waits for a question |

Example session:

```text
==================================================
       OFFLINE GENAI CHATBOT
==================================================
Ask any question.
Type /reset to clear history.
Type /exit to stop.
==================================================

You: What is Generative AI?
Bot: Generative AI is a type of artificial intelligence that can create
new content, such as text, images, music, or code...

You: /reset
Bot: Conversation history cleared.

You: /exit
Bot: Goodbye!
```

*Example response is illustrative; actual output depends on model generation.*

## Configuration

The notebook defines these settings in the response-generation cell:

```python
MAX_NEW_TOKENS = 120
MAX_HISTORY_MESSAGES = 6
```

- `MAX_NEW_TOKENS` controls the maximum number of newly generated tokens per response.
- `MAX_HISTORY_MESSAGES` controls how many recent messages are included when preparing the model input.

The interactive chat function also trims stored history to the most recent 12 message entries. Because each exchange normally contains a user message and an assistant message, this is approximately six recent exchanges. The history exists in memory for the current session and is cleared when `/reset` is used or the program is restarted.

Generation settings such as `temperature`, `top_p`, and `repetition_penalty` are configured in `model.generate()`.

## Limitations

- The model is relatively small and may produce incomplete, inaccurate, or outdated answers.
- It does not automatically browse the web or retrieve live information.
- Conversation history is held in memory; it is not saved to a database or restored after restarting.
- CPU inference may be slower than inference on supported GPU hardware.
- “Offline” applies after dependencies and model files have been installed/downloaded. The initial setup needs internet access.
- The chatbot does not guarantee factual accuracy. Verify important information independently.

## Troubleshooting

### Error: model directory not found

Run the model-download cell first and check that `smollm2_model/` is in the expected working directory.

### Error: missing Python package

Install the required package in the same Python environment used by Jupyter, then restart the kernel if necessary.

### Chatbot is slow

CPU generation speed varies with hardware. Close other memory-intensive applications and consider reducing `MAX_NEW_TOKENS` for shorter responses.

### Responses do not use previous context

Ensure you pass the `history` argument to `generate_response()` and append each user message and assistant response to the history list.

## Technologies Used

- **Python** — chatbot logic and command-line interaction
- **PyTorch** — model inference
- **Hugging Face Transformers** — tokenizer, model loading, chat template, and text generation
- **SmolLM2-360M-Instruct** — locally stored instruction-tuned language model
- **Jupyter Notebook** — development and execution environment

## Possible Future Improvements

- Add a Streamlit or Flask user interface.
- Save and restore conversations from a local file or database.
- Add configurable model and generation settings.
- Add streaming responses and a typing indicator.
- Add conversation export and improved error handling.

## License and Model Attribution

This repository's code and the model have separate licensing terms. Before redistributing the model files or publishing a packaged copy, review the license and usage terms provided by the [SmolLM2 model repository](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) and any dependencies.

## Author

**Aniruddh Chougule**

Add your GitHub profile or repository link here:

- GitHub: `<your-github-profile-url>`
- Project Repository: `<your-repository-url>`
