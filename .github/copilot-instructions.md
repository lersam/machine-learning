# Copilot Instructions for AI Agents
You are an expert in deep learning, transformers, diffusion models, and LLM development, with a focus on Python libraries such as PyTorch, Diffusers, Transformers, and Gradio.

## Project Overview
This workspace contains multiple machine learning projects, each in its own directory or notebook. The codebase is organized by topic and use case, with a focus on clustering, time series forecasting, and number recognition.

## Key Components
- **Notebooks at root**: Entry points for experiments and tutorials (e.g., `k_means_clustering.ipynb`, `cluster_id_assignment.ipynb`).
- **autoencoder_lstm/energy_consumption/**: Time series forecasting using LSTM autoencoders. Main logic in `main.py`, with supporting modules in `support/`.
- **number-recognition/**: Handwritten digit recognition using PyTorch. Contains model code, training, and prediction notebooks. Models are stored in `local_data/models/`.
- **old/Matrix-Properties/**: Legacy or reference notebooks for matrix operations.

## Patterns & Conventions
- **Support modules**: Each major subproject has a `support/` or `support.py` for utilities and shared logic.
- **Notebooks for workflow**: Most experimentation and results are in Jupyter notebooks. Scripts like `main.py` are for batch or production runs.
- **Requirements**: Each subproject may have its own `requirements.txt`. Install dependencies per subproject as needed.
- **Data/Models**: Models and data are stored locally within each subproject (e.g., `number-recognition/local_data/models/`).

## Developer Workflows
- **Run experiments**: Open the relevant notebook and execute cells interactively.
- **Production runs**: Use scripts like `autoencoder_lstm/energy_consumption/main.py`.
- **Dependencies**: Install with `pip install -r requirements.txt` in the relevant subproject directory.
- **Model files**: Models are saved/loaded from local subdirectories; paths are relative to each subproject.

## Integration & External Dependencies
- **PyTorch**: Used for deep learning in number recognition.
- **Scikit-learn**: Used for clustering and classic ML tasks.
- **Jupyter**: Primary interface for development and results.

## Examples
- To run LSTM forecasting: `python autoencoder_lstm/energy_consumption/main.py`
- To train or test number recognition: Use `training_section.ipynb` or `prediction_section.ipynb` in `number-recognition/`.

## Tips for AI Agents
- Prefer editing or creating code in the relevant subproject directory.
- When adding dependencies, update the correct `requirements.txt`.
- Follow the pattern of using `support/` modules for reusable code.
- Reference model/data paths relative to the subproject, not the workspace root.

---
For more details, see the notebooks and `README.md` at the project root.
