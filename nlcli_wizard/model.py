"""
Model Manager - Handles SLM loading, inference, and model downloads
"""

from typing import Optional, Dict, Any
from pathlib import Path
import json
import os

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False

try:
    from huggingface_hub import hf_hub_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False


class ModelManager:
    """
    Manages loading and inference of the fine-tuned SLM.

    Handles:
    - Lazy model loading (only when first needed)
    - Model downloading from HuggingFace Hub
    - Inference with llama-cpp-python
    - Model caching
    """

    DEFAULT_MODEL_REPO = "YOUR_USERNAME/nlcli-tinyllama-venvy"  # Update after training
    DEFAULT_MODEL_FILE = "nlcli-tinyllama-venvy-q4_k_m.gguf"

    def __init__(
        self,
        cli_tool: str,
        model_path: Optional[Path] = None,
        n_ctx: int = 512,
        n_threads: int = 4,
    ):
        """
        Initialize model manager.

        Args:
            cli_tool: Name of CLI tool (used to find appropriate model)
            model_path: Path to local model file (downloads if None)
            n_ctx: Context window size
            n_threads: Number of CPU threads for inference
        """
        self.cli_tool = cli_tool
        self.n_ctx = n_ctx
        self.n_threads = n_threads

        # Lazy loading
        self._model: Optional[Llama] = None
        self._model_path = model_path

        # Model cache directory
        self.cache_dir = Path.home() / ".cache" / "nlcli-wizard" / "models"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def model(self) -> Llama:
        """
        Lazy-load the model on first access.
        """
        if self._model is None:
            self._load_model()
        return self._model

    def _load_model(self):
        """
        Load the quantized GGUF model using llama-cpp-python.
        """
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError(
                "llama-cpp-python not installed. "
                "Install with: pip install llama-cpp-python"
            )

        # Determine model path
        if self._model_path is None:
            self._model_path = self._download_model()

        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self._model_path}. "
                "Run training pipeline or download pre-trained model."
            )

        print(f"Loading model from {self._model_path}...")

        # Load with llama.cpp
        self._model = Llama(
            model_path=str(self._model_path),
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            n_gpu_layers=0,  # CPU only (set >0 if you have GPU)
            verbose=False,
        )

        print("Model loaded successfully!")

    def _download_model(self) -> Path:
        """
        Download model from HuggingFace Hub if not cached.

        Returns:
            Path to downloaded model file
        """
        if not HF_HUB_AVAILABLE:
            raise ImportError(
                "huggingface-hub not installed. "
                "Install with: pip install huggingface-hub"
            )

        # Check cache first
        cached_model = self.cache_dir / self.DEFAULT_MODEL_FILE
        if cached_model.exists():
            print(f"Using cached model: {cached_model}")
            return cached_model

        print(f"Downloading model from {self.DEFAULT_MODEL_REPO}...")

        try:
            downloaded_path = hf_hub_download(
                repo_id=self.DEFAULT_MODEL_REPO,
                filename=self.DEFAULT_MODEL_FILE,
                cache_dir=str(self.cache_dir),
            )

            print(f"Model downloaded to {downloaded_path}")
            return Path(downloaded_path)

        except Exception as e:
            raise RuntimeError(
                f"Failed to download model: {e}\n"
                "Make sure you have:\n"
                "1. Trained and uploaded the model to HuggingFace Hub\n"
                "2. Set the correct repo ID in ModelManager.DEFAULT_MODEL_REPO\n"
                "3. Or provide a local model path explicitly"
            )

    def generate_command(self, natural_language: str) -> str:
        """
        Generate CLI command from natural language using the SLM.

        Args:
            natural_language: User's natural language instruction

        Returns:
            Model output (to be parsed by NLCLIAgent)
        """
        # Build prompt (matches training format)
        prompt = self._build_prompt(natural_language)

        # Generate with model
        output = self.model(
            prompt,
            max_tokens=128,
            temperature=0.1,  # Low temperature for more deterministic output
            top_p=0.9,
            stop=["</s>", "\n\n"],  # Stop tokens
            echo=False,
        )

        # Extract generated text
        generated = output["choices"][0]["text"].strip()

        return generated

    def _build_prompt(self, natural_language: str) -> str:
        """
        Build the prompt for the model (must match training format).

        Training format:
        <s>[INST] Translate to {cli_tool} command: {natural_language} [/INST]
        COMMAND: {command}
        CONFIDENCE: {confidence}
        EXPLANATION: {explanation}
        </s>
        """
        prompt = f"<s>[INST] Translate to {self.cli_tool} command: {natural_language} [/INST]\n"

        return prompt

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model metadata
        """
        if self._model_path is None or not self._model_path.exists():
            return {
                "status": "not_loaded",
                "path": None,
                "size_mb": 0,
            }

        size_mb = self._model_path.stat().st_size / (1024 * 1024)

        return {
            "status": "loaded" if self._model is not None else "available",
            "path": str(self._model_path),
            "size_mb": round(size_mb, 2),
            "cli_tool": self.cli_tool,
        }


class ModelTrainingConfig:
    """
    Configuration for model training pipeline.

    This is used in the Colab training notebook.
    """

    def __init__(
        self,
        base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        dataset_path: Path = Path("assets/data/training_examples.jsonl"),
        output_dir: Path = Path("assets/models/trained"),
        num_epochs: int = 3,
        learning_rate: float = 2e-4,
        batch_size: int = 4,
        max_seq_length: int = 512,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ):
        """
        Initialize training configuration.

        Args:
            base_model: HuggingFace model ID
            dataset_path: Path to training JSONL file
            output_dir: Where to save trained model
            num_epochs: Number of training epochs
            learning_rate: Learning rate for fine-tuning
            batch_size: Training batch size
            max_seq_length: Maximum sequence length
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout rate
        """
        self.base_model = base_model
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "base_model": self.base_model,
            "dataset_path": str(self.dataset_path),
            "output_dir": str(self.output_dir),
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "max_seq_length": self.max_seq_length,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
        }

    def save(self, path: Path):
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ModelTrainingConfig":
        """Load config from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)

        return cls(
            base_model=data["base_model"],
            dataset_path=Path(data["dataset_path"]),
            output_dir=Path(data["output_dir"]),
            num_epochs=data["num_epochs"],
            learning_rate=data["learning_rate"],
            batch_size=data["batch_size"],
            max_seq_length=data["max_seq_length"],
            lora_r=data["lora_r"],
            lora_alpha=data["lora_alpha"],
            lora_dropout=data["lora_dropout"],
        )
