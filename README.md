# MDocAgent

A multimodal, multi-agent pipeline for answering questions about PDF documents. MDocAgent retrieves relevant text and page images, generates complementary responses through specialized agents, and combines them into a final answer.

This README documents the implementation in the `mdoc` directory. Installation scripts are located in its parent directory. Commands below explicitly identify which directory to use.

> **Implementation status:** The current code requires dependency and model setup. The standard entry point has known issues, so the usage example below calls the core pipeline directly. These instructions are based on source inspection; GPU inference and dependency compatibility have not been verified end to end.

## Contents

- [Overview](#overview)
- [Requirements and installation](#requirements-and-installation)
- [Download models](#download-models)
- [Run PDF question answering](#run-pdf-question-answering)
- [Configuration](#configuration)
- [File reference](#file-reference)
- [Outputs and caching](#outputs-and-caching)
- [Known limitations and troubleshooting](#known-limitations-and-troubleshooting)

## Overview

```text
PDF + question
    |
    v
Extract text and render page images with PyMuPDF
    |
    +--> Text retrieval: ColBERT / RAGatouille
    |
    +--> Image retrieval: ColPali
    |
    v
Load retrieved text and image references
    |
    v
General agent: initial answer and reflection
    |
    +--> Text agent: answer using text and reflection clues
    |
    +--> Image agent: answer using images and reflection clues
    |
    v
Summary agent: synthesize the final answer
```

The current pipeline processes one PDF and one question at a time. Agent calls execute sequentially. This directory contains inference code; it does not include a web interface, a model-serving launcher, a training entry point, or a complete batch evaluation workflow.

The dataset loader includes text fallback readers for several formats, but the main extraction pipeline is designed for PDFs.

## Requirements and installation

The existing repository setup uses Python 3.10 and a Linux CUDA environment. Several components explicitly select CUDA devices, so CPU-only execution is not supported without code changes. Native Windows compatibility, particularly for RAGatouille/ColBERT dependencies, requires additional validation.

From the **parent of the `mdoc` directory**, create an environment and choose one installation script:

```bash
conda create -n mdocagent python=3.10
conda activate mdocagent
bash install_dev.sh
```

The repository provides two alternative scripts:

| Script | PyTorch configuration |
| --- | --- |
| `install.sh` | `torch==2.4` |
| `install_dev.sh` | `torch==2.7.1+cu128`, matching torchvision/torchaudio versions, and the CUDA 12.8 wheel index |

Both scripts install a pinned Transformers commit and dependencies including Accelerate, Qwen VL utilities, PyMuPDF, pandas, the OpenAI client, Hydra, `colpali_engine==0.1.1`, `mteb==1.2`, RAGatouille, and PyPDF2.

These are repository-supplied versions, not a verified compatibility matrix. In particular, the local Qwen adapter requires a Transformers build that exposes `Qwen2_5_VLForConditionalGeneration`.

The source also directly imports `requests`, `Pillow`, `tqdm`, `omegaconf`, `huggingface_hub`, and `peft`; ensure they are installed. The optional DOCX and HTML fallback readers require `python-docx` and `beautifulsoup4`, respectively.

There is no `requirements.txt`, `pyproject.toml`, or `setup.py` in this directory. An editable package installation is therefore not provided here.

## Download models

From the **`mdoc` directory containing this README**, run:

```bash
python scripts/download_models.py
```

The script downloads every resource in the following table, including the 32B generation model. It has no model-selection argument. If you only need local 7B generation, adjust its download list before running it to omit unused generation models. All three retrieval resources remain necessary for the current pipeline.

| Hugging Face repository | Destination relative to `mdoc/` | Purpose |
| --- | --- | --- |
| `vidore/colpaligemma-3b-mix-448-base` | `retrieval/local_colpaligemma/` | Image retrieval base model |
| `vidore/colpali` | `retrieval/local_colpali_adapter/` | Image retrieval adapter and processor |
| `colbert-ir/colbertv2.0` | `retrieval/local_colbert/` | Text retrieval model |
| `Qwen/Qwen2.5-VL-7B-Instruct` | `models/qwen2.5-vl-7b/` | Default local Qwen generation configuration |
| `Qwen/Qwen2.5-VL-32B-Instruct` | `models/qwen2.5-vl-32b/` | Optional local generation weights; not selected by the default offline configuration |
| `tiiuae/falcon-rw-1b` | `models/falcon-rw-1b/` | Local text-only Falcon generation |

Downloads require network access and any applicable repository permissions. Complete them in a separate process before starting inference: importing the retrieval modules sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.

## Run PDF question answering

### Call the core pipeline

Create `run_example.py` in the **`mdoc` directory**, alongside this README:

```python
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent

# Select available GPUs before importing retrieval modules or PyTorch.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, str(ROOT))

from hydra import compose, initialize_config_dir
from scripts.main_core import run_core

with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
    cfg = compose(config_name="setting")
    cfg.input.pdf_path = str(Path("/absolute/path/example.pdf").resolve())
    cfg.input.question = "What are the main findings of this document?"
    cfg.extract.extract_dir = str(ROOT / "tmp")
    cfg.mdoc_agent.cuda_visible_devices = "0"

    # Use local Qwen2.5-VL 7B for all agents, including the summary agent.
    for agent_config in cfg.mdoc_agent.agents:
        agent_config.model = "qwen25vloffline"
    cfg.mdoc_agent.sum_agent.model = "qwen25vloffline"

    answer = run_core(cfg)
    print("Final answer:", answer)
```

Replace the PDF path with an existing document. On Windows, a path can be written as `Path(r"C:\data\example.pdf")`.

Run from the same directory:

```bash
python run_example.py
```

This example requires the retrieval resources, local Qwen weights, compatible dependencies, and sufficient GPU memory. It selects local generation because that adapter processes actual image inputs. The example script is supplied above for you to create; it is not included as a separate file.

`run_core()` replaces model and agent names in the configuration with composed configuration objects. Compose a fresh configuration for each subsequent call.

### Existing entry point

`scripts/main.py` exposes `main(file_path, question_string)`, but currently has two issues:

1. Its import uses `mdoc.mdoc.scripts.main_core`, which does not match the conventional package layout shown here.
2. Its `__main__` block supplies a hard-coded PDF path and question. The wrapper then overwrites the corresponding Hydra configuration values, including command-line overrides.

To retain the Python wrapper, change its import to `from .main_core import run_core`, then import `main` using `from mdoc.scripts.main import main` from the parent directory. Select an appropriate generation backend in `config/setting.yaml` before calling it.

For a Hydra command-line interface, create or adapt an entry point that passes `cfg` directly to `run_core(cfg)` without overwriting the input fields. The original parent README command requires these issues to be addressed first.

### Default HTTP backend

The default `qwen25vl` configuration uses `models/qwen.py`, which sends requests to:

```text
http://localhost:40911/v1/chat/completions
```

You must provide a compatible model service separately. Downloading weights does not launch it, and this directory does not include the server implementation.

The current HTTP adapter sends text and `<image>` placeholders without image bytes or image URLs. A successful HTTP response therefore does not establish that the backend received the page images. Use or adapt an image-capable backend for visual question answering; the core example above uses the local Qwen adapter.

## Configuration

The main configuration is [`config/setting.yaml`](config/setting.yaml).

| Setting | Default | Description |
| --- | --- | --- |
| `input.pdf_path` | Empty string | Input PDF path |
| `input.question` | Empty string | Question about the document |
| `extract.extract_dir` | `./mdoc/tmp/` | Extraction cache root, relative to the working directory |
| `extract.max_page` | `1000` | Maximum number of pages to extract |
| `retrieval.top_k` | `5` | Number of results selected by each retriever |
| `text_retrieval.text_max_character_per_page` | `1024` | Per-page character limit for text indexing |
| `image_retrieval.batch_size` | `2` | Image embedding batch size |
| `predict.top_k` | `2` | Maximum retrieved entries passed to agents per modality |
| `predict.max_page` | `1000` | Page loading limit |
| `predict.max_character_per_page` | `100000` | Per-page character limit when loading agent references |
| `predict.use_mix` | `false` | Read a mixed ranking if supplied; the current pipeline does not generate one |
| `mdoc_agent.cuda_visible_devices` | `'0,1,2,3'` | GPU visibility set before prediction, after retrieval may already initialize CUDA |
| `mdoc_agent.ans_key` | `ans_key` | Sample field containing the final answer |
| `mdoc_agent.save_message` | `false` | Retain summary-model messages in the sample; does not write them to disk |

Set GPU visibility before importing the pipeline, as shown in the example, rather than relying solely on the late assignment inside `run_core()`.

Some experiment-related settings, including `eval_agent`, `run-name`, `save_freq`, and `truncate_len`, are not fully used by the current single-sample workflow. `save_freq` does not enable automatic checkpointing here, and `max_retries` has no corresponding general retry implementation.

## File reference

Paths in this section are relative to the `mdoc` directory.

### Entry points and configuration processing

| File | Responsibility |
| --- | --- |
| `scripts/main.py` | Hydra wrapper that prints and returns an answer; see the entry-point limitations above |
| `scripts/main_core.py` | `run_core(cfg)`: extraction, retrieval, agent configuration composition, and prediction |
| `scripts/download_models.py` | Downloads retrieval and generation resources; all downloads execute at module level |
| `config_processer/cfg_processer.py` | `ConfigProcesser` and page content structure; validates file existence, builds cache paths, maintains the sample, and loads extracted pages |
| `config_processer/__init__.py` | Exports `ConfigProcesser` |

### Extraction, retrieval, and data loading

| File | Responsibility |
| --- | --- |
| `extract/extract_file.py` | Extracts page text and renders PNG files at 144 DPI by default; skips files that already exist |
| `extract/__init__.py` | Exports `extract` |
| `retrieval/base_retrieval.py` | Abstract retrieval interface with `prepare()` and `find_top_k()` |
| `retrieval/text_retrieval.py` | Loads local ColBERT, builds an index, and stores retrieved page indices and scores |
| `retrieval/image_retrieval.py` | Loads local ColPali and its PEFT adapter, encodes pages and queries, and ranks images |
| `retrieval/__init__.py` | Exports both retrievers; importing them also enables their offline environment settings |
| `mydatasets/base_dataset.py` | Loads retrieved text and image paths, limits text length, and provides fallback text readers |
| `predict/predict_sample.py` | Creates `BaseDataset` and `MDocAgent`, runs one prediction, and returns the updated sample |
| `predict/__init__.py` | Exports `predict` |

### Agents

| File | Responsibility |
| --- | --- |
| `agents/base_agent.py` | Dynamically loads models, applies prompts and modality filters, manages conversation history, performs reflection, and provides answer evaluation helpers |
| `agents/multi_agent_system.py` | Creates agents, shares model instances, summarizes responses, predicts a single sample, and clears history |
| `agents/mdoc_agent.py` | Implements the general-answer, reflection, text-answer, image-answer, and synthesis sequence |
| `agents/ablations.py` | Defines `MDAi`, `MDAt`, and `MDAs` ablation variants; their interfaces require updating before use |

Keep the configured agent order as **image, text, general**: the implementation accesses them by indices `0`, `1`, and `-1`.

Model instances are cached by `class_name`, so different configurations using the same class name may unintentionally share one model.

### Model adapters

| File | Responsibility and requirements |
| --- | --- |
| `models/base_model.py` | Common prediction interface, message assembly, and CUDA cache cleanup |
| `models/qwen.py` | HTTP Qwen adapter; requires a separate local service and currently sends image placeholders only |
| `models/qwenoffline.py` | Local Qwen2.5-VL loading from `models/<model_id>`; processes actual images with `qwen_vl_utils` and requires CUDA |
| `models/falcon.py` | Loads Falcon RW 1B by model ID for text generation; fixes the pipeline device to GPU 0 |
| `models/falconoffline.py` | Loads local Falcon RW 1B from `models/<model_id>`; text-only |
| `models/openai.py` | API adapter with Base64 image messages; requires credentials, network access, and an available model |

Using an API generation adapter does not bypass the local retrieval stages. Falcon does not implement image message creation and needs additional adaptation for this multimodal workflow.

### Legacy agent implementations

The default workflow imports the top-level `agents/` package. The following files under `models/agents/` contain older implementations:

| File | Responsibility and compatibility notes |
| --- | --- |
| `models/agents/base_agent.py` | Older agent wrapper without the top-level version's image-path string conversion |
| `models/agents/mdoc_agent.py` | Older cooperative inference implementation; its constructor does not match the current top-level parent class |
| `models/agents/multi_agent_system.py` | Batch prediction, resume, and periodic saving logic; calls dataset methods such as `dump_reults()` that the current dataset does not implement |
| `models/agents/ablations.py` | Older ablation variants requiring interface updates |

These files are not ready-to-use batch entry points for the current dataset implementation.

### Configuration files

| File | Responsibility |
| --- | --- |
| `config/setting.yaml` | Main extraction, retrieval, agent, prediction, and input settings |
| `config/agent/base.yaml` | Agent class path, modality switches, common settings, and evaluation prompt |
| `config/agent/image_agent.yaml` | Image-only agent prompt |
| `config/agent/text_agent.yaml` | Text-only agent prompt |
| `config/agent/general_agent.yaml` | Multimodal agent and reflection prompts |
| `config/agent/sum_agent.yaml` | Synthesis prompt requesting a dictionary with an `Answer` field |
| `config/model/base.yaml` | Shared settings: 256 generated tokens and temperature 0; model identifiers are placeholders |
| `config/model/qwen25vl.yaml` | Default HTTP adapter configuration with the Qwen2.5-VL 32B model ID |
| `config/model/qwen25vloffline.yaml` | Local Qwen2.5-VL 7B configuration |
| `config/model/qwen2vl.yaml` | References `Qwen2VL`, a class absent from the current adapter |
| `config/model/falcon.yaml` | Falcon RW 1B model-ID configuration |
| `config/model/falconoffline.yaml` | Local Falcon RW 1B configuration |
| `config/model/openai.yaml` | API model configuration, currently naming `gpt-4o`, with an empty API key field |

The remaining `__init__.py` files in the directory root, `scripts/`, `config/`, `agents/`, `models/`, `models/agents/`, and `mydatasets/` are empty package markers.

## Outputs and caching

- **Extracted pages:** `<extract.extract_dir>/<document_stem>/<document_stem>_0.png` and matching `.txt` files. Page numbering starts at zero.
- **Text indexes:** The code constructs paths under `retrieval/local_colbert/indexes/`. The path returned by RAGatouille and stored in the sample determines the actual index location.
- **Sample fields:** PDF path, question, document ID, retrieved page indices and scores, and the final answer under `ans_key`.
- **Return value:** `run_core(cfg)` returns the final answer and prints the debug sample. It does not automatically save an answer JSON file.

The default extraction path is relative to the working directory. Launching from inside `mdoc/` with the unchanged default produces a nested `mdoc/tmp/` beneath it. The example instead uses an absolute path to `tmp/` beside this README.

Extraction caches are keyed by the document filename stem. Different PDFs with the same stem can reuse the same cache, and existing page files are skipped. Use a distinct cache directory or remove the corresponding stale cache when replacing a document.

## Known limitations and troubleshooting

| Symptom | What to check |
| --- | --- |
| `No module named mdoc.mdoc` | Use the core example or fix the wrapper import and launch context |
| Missing `./files/example.pdf` or ignored input overrides | The existing entry point overwrites Hydra inputs with hard-coded arguments |
| Missing local model files | Verify download destinations; retrieval modules enforce offline loading |
| Missing `Qwen2VL` | `qwen2vl.yaml` references a class not implemented in `models/qwen.py` |
| Missing `Qwen2_5_VLForConditionalGeneration` | Check Transformers compatibility with the local Qwen adapter |
| CUDA unavailable, device mismatch, or out-of-memory errors | Several paths assume CUDA; configure visible GPUs before imports and reduce page counts or image batch size as appropriate |
| Failed text indexing or empty retrieval results | Check logs, dependencies, extracted text, and `pid_docid_map.json`; some exceptions are printed and converted into empty results |
| Empty text from scanned pages | `page.get_text()` does not perform OCR; visual understanding depends on downstream image-capable models |
| Text encoding errors on Windows | Extraction writes text without an explicit encoding, while loaders read UTF-8; update the writer to `open(txt_file, "w", encoding="utf-8")` |
| Generic failure text after an HTTP connection error | The HTTP adapter catches request errors and returns a fixed failure message; this is not a successful document answer |
| Falcon fails on image inputs or later reflection calls | Image message creation and conversation history require adaptation |
| Batch evaluation or ablation errors | Legacy methods and constructor/prediction signatures do not match the current dataset and agent interfaces |

Additional implementation details:

- Text retrieval takes top passage hits without deduplicating their page indices. Multiple hits from one page can occupy multiple result slots.
- ColPali scoring depends on the evaluator API and its returned score structure; validate it against the installed version.
- The OpenAI adapter labels image data as JPEG, while extraction produces PNG files. Align the MIME type with the actual image format before relying on that adapter.
- `save_message` retains summary-model messages rather than a complete independent trace of every agent stage.
- Evaluation helpers reference dataset methods such as `load_latest_results()` that are not implemented in the current `BaseDataset`.
