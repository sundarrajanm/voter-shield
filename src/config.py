"""
Centralized configuration management.

Configuration is loaded from:
1. Environment variables
2. .env file (if present)
3. Default values

Usage:
    from src.config import Config
    config = Config()
    print(config.debug)  # True if DEBUG=1 in environment
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


def _load_dotenv(dotenv_path: Optional[Path] = None) -> None:
    """
    Minimal .env loader (no external dependency).
    
    Supports KEY=VALUE, ignores blank lines and comments (#).
    Does not override existing environment variables.
    """
    if dotenv_path is None:
        dotenv_path = Path(__file__).resolve().parent.parent / ".env"
    
    if not dotenv_path.exists() or not dotenv_path.is_file():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        # Only set if not already in environment
        if os.getenv(key) in (None, ""):
            os.environ[key] = value


# Load .env on module import
_load_dotenv()


def _get_bool_env(key: str, default: bool = False) -> bool:
    """Get boolean from environment variable."""
    value = os.getenv(key, "").strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    return default


def _get_int_env(key: str, default: int) -> int:
    """Get integer from environment variable."""
    value = os.getenv(key, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_float_env(key: str, default: Optional[float] = None) -> Optional[float]:
    """Get float from environment variable."""
    value = os.getenv(key, "").strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


@dataclass
class ROIConfig:
    """Region of Interest configuration for image cropping."""
    x1: float
    y1: float
    x2: float
    y2: float
    
    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class AIConfig:
    """AI/LLM configuration for metadata extraction."""
    provider: str = field(default_factory=lambda: os.getenv("AI_PROVIDER", "Groq"))
    api_key: str = field(default_factory=lambda: os.getenv("AI_API_KEY", ""))
    model: str = field(default_factory=lambda: os.getenv("AI_MODEL", "meta-llama/llama-4-maverick-17b-128e-instruct"))
    base_url: str = field(default_factory=lambda: os.getenv("AI_BASE_URL", "https://api.groq.com/openai/v1/chat/completions"))
    timeout_sec: int = field(default_factory=lambda: _get_int_env("AI_TIMEOUT_SEC", 120))
    response_format: str = field(
        default_factory=lambda: os.getenv("AI_RESPONSE_FORMAT", "").strip().lower()
    )
    batch_size: int = field(default_factory=lambda: _get_int_env("AI_OCR_BATCH_SIZE", 5))
    id_batch_size: int = field(default_factory=lambda: _get_int_env("AI_ID_BATCH_SIZE", 5))
    
    # Retry configuration
    max_retries: int = field(default_factory=lambda: _get_int_env("AI_MAX_RETRIES", 3))
    retry_delay_sec: float = field(default_factory=lambda: _get_float_env("AI_RETRY_DELAY_SEC", 2.0) or 2.0)

    
    # Cost tracking
    input_cost_per_1m_usd: Optional[float] = field(
        default_factory=lambda: _get_float_env("AI_INPUT_COST_PER_1M_USD")
    )
    output_cost_per_1m_usd: Optional[float] = field(
        default_factory=lambda: _get_float_env("AI_OUTPUT_COST_PER_1M_USD")
    )
    cost_currency: str = field(
        default_factory=lambda: os.getenv("AI_COST_CURRENCY", "USD").upper()
    )
    
    # Alias properties for backward compatibility
    @property
    def input_cost_per_1m(self) -> Optional[float]:
        """Alias for input_cost_per_1m_usd."""
        return self.input_cost_per_1m_usd
    
    @property
    def output_cost_per_1m(self) -> Optional[float]:
        """Alias for output_cost_per_1m_usd."""
        return self.output_cost_per_1m_usd
    
    @property
    def has_pricing(self) -> bool:
        return self.input_cost_per_1m_usd is not None or self.output_cost_per_1m_usd is not None
    
    def get_normalized_base_url(self) -> str:
        """
        Get normalized base_url for OpenAI SDK.
        
        The OpenAI SDK expects a *base* URL without the endpoint.
        For Gemini, converts full endpoints to base URL format.
        """
        u = (self.base_url or "").strip()
        if not u:
            # Use default based on provider
            if self.provider.lower() == "gemini":
                return "https://generativelanguage.googleapis.com/v1beta/openai/"
            return ""  # OpenAI SDK will use default
        
        u = u.rstrip("/")
        if u.endswith("/chat/completions"):
            u = u[: -len("/chat/completions")]
        
        # Keep trailing slash for consistency
        return u.rstrip("/") + "/"
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> Optional[float]:
        """Estimate cost in USD for given token counts."""
        if not self.has_pricing:
            return None
        
        cost = 0.0
        if self.input_cost_per_1m_usd:
            cost += (input_tokens / 1_000_000) * self.input_cost_per_1m_usd
        if self.output_cost_per_1m_usd:
            cost += (output_tokens / 1_000_000) * self.output_cost_per_1m_usd
        return cost


@dataclass
class S3Config:
    """AWS S3 configuration for remote file access."""
    # AWS credentials (can also use IAM roles, AWS_PROFILE, etc.)
    access_key_id: str = field(default_factory=lambda: os.getenv("AWS_ACCESS_KEY_ID", ""))
    secret_access_key: str = field(default_factory=lambda: os.getenv("AWS_SECRET_ACCESS_KEY", ""))
    session_token: str = field(default_factory=lambda: os.getenv("AWS_SESSION_TOKEN", ""))
    
    # Region
    region: str = field(default_factory=lambda: os.getenv("AWS_REGION", "ap-south-1"))
    
    # Default bucket (optional, can be specified per URL)
    default_bucket: str = field(default_factory=lambda: os.getenv("S3_DEFAULT_BUCKET", ""))
    
    # Download settings
    download_dir: str = field(default_factory=lambda: os.getenv("S3_DOWNLOAD_DIR", ""))
    
    # Connection settings
    connect_timeout: int = field(default_factory=lambda: _get_int_env("S3_CONNECT_TIMEOUT", 10))
    read_timeout: int = field(default_factory=lambda: _get_int_env("S3_READ_TIMEOUT", 60))
    max_retries: int = field(default_factory=lambda: _get_int_env("S3_MAX_RETRIES", 3))
    
    @property
    def has_credentials(self) -> bool:
        """Check if explicit credentials are configured."""
        return bool(self.access_key_id and self.secret_access_key)
    
    @property
    def is_configured(self) -> bool:
        """Check if S3 is configured (credentials or IAM role assumed)."""
        # Returns True if credentials exist or we assume IAM role will be used
        return self.has_credentials or bool(os.getenv("AWS_PROFILE"))


@dataclass
class OCRConfig:
    """OCR (Tesseract) configuration."""
    languages: str = field(default_factory=lambda: os.getenv("OCR_LANGUAGES", "eng+tam"))
    tesseract_path: str = field(default_factory=lambda: os.getenv("TESSERACT_PATH", ""))
    allow_next_line: bool = field(default_factory=lambda: _get_bool_env("OCR_ALLOW_NEXT_LINE", True))
    
    # ROI configurations (relative coordinates 0-1)
    epic_roi: ROIConfig = field(default_factory=lambda: ROIConfig(0.449227, 0.009029, 0.839956, 0.162528))
    serial_roi: ROIConfig = field(default_factory=lambda: ROIConfig(0.152318, 0.002257, 0.373068, 0.160271))
    house_roi: ROIConfig = field(default_factory=lambda: ROIConfig(0.302428, 0.415350, 0.665563, 0.553047))
    deleted_roi: ROIConfig = field(default_factory=lambda: ROIConfig(0.199338, 0.146727, 0.763797, 0.823928))

@dataclass
class MergeConfig:
    """Image merging configuration."""
    batch_size: int = field(default_factory=lambda: _get_int_env("MERGE_BATCH_SIZE", 10))


@dataclass
class TopMergeConfig:
    """Crop-top image merging configuration."""
    batch_size: int = field(default_factory=lambda: _get_int_env("TOP_MERGE_BATCH_SIZE", 10))


@dataclass
class IDCropConfig:
    """ID field crop and merge configuration."""
    batch_size: int = field(default_factory=lambda: _get_int_env("ID_CROP_BATCH_SIZE", 10))


@dataclass
class DBConfig:
    """Database configuration (PostgreSQL)."""
    host: str = field(default_factory=lambda: os.getenv("DB_HOST", ""))
    port: int = field(default_factory=lambda: _get_int_env("DB_PORT", 5432))
    name: str = field(default_factory=lambda: os.getenv("DB_NAME", ""))
    user: str = field(default_factory=lambda: os.getenv("DB_USER", ""))
    password: str = field(default_factory=lambda: os.getenv("DB_PASSWORD", ""))
    schema: str = field(default_factory=lambda: os.getenv("DB_SCHEMA", "public"))
    ssl_mode: str = field(default_factory=lambda: os.getenv("DB_SSL_MODE", "prefer"))
    required: bool = field(default_factory=lambda: _get_bool_env("DB_REQUIRED", False))
    
    @property
    def is_configured(self) -> bool:
        """Check if minimal DB config is present."""
        return bool(self.host and self.name and self.user)


@dataclass
class CropConfig:
    """Image cropping configuration."""
    # Canonical size for detection
    canon_width: int = 1187
    canon_height: int = 1679
    
    # Box filters
    min_box_area_frac: float = 0.006
    max_box_area_frac: float = 0.25
    min_aspect_ratio: float = 0.55
    max_aspect_ratio: float = 2.8
    
    # Padding
    padding: int = 3
    
    # Grid line detection
    hline_scale: int = 25
    vline_scale: int = 25
    
    # Diagram filtering
    diagram_filter_mode: str = field(
        default_factory=lambda: os.getenv("CROP_DIAGRAM_FILTER", "auto")
    )  # auto, on, off


@dataclass
class Config:
    """
    Main application configuration.
    
    All settings are loaded from environment variables with sensible defaults.
    Set DEBUG=1 in environment to enable debug mode.
    """
    
    # Base directory (project root)
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    
    # Directory paths
    pdfs_dir: Path = field(default=None)
    extracted_dir: Path = field(default=None)
    logs_dir: Path = field(default=None)
    
    # Debug mode (enables verbose logging, raw OCR dumps, ROI overlays)
    debug: bool = field(default_factory=lambda: _get_bool_env("DEBUG", False))
    
    # PDF extraction
    render_dpi: int = field(default_factory=lambda: _get_int_env("RENDER_DPI", 200))
    
    # Sub-configurations
    ai: AIConfig = field(default_factory=AIConfig)
    s3: S3Config = field(default_factory=S3Config)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    crop: CropConfig = field(default_factory=CropConfig)
    merge: MergeConfig = field(default_factory=MergeConfig)
    top_merge: TopMergeConfig = field(default_factory=TopMergeConfig)
    id_crop: IDCropConfig = field(default_factory=IDCropConfig)
    db: DBConfig = field(default_factory=DBConfig)
    
    # Processing limits
    default_limit: int = field(default_factory=lambda: _get_int_env("DEFAULT_LIMIT", 0))
    default_page_limit: int = field(default_factory=lambda: _get_int_env("DEFAULT_PAGE_LIMIT", 0))
    
    def __post_init__(self):
        """Resolve paths after initialization."""
        # Set default paths relative to base_dir
        if self.pdfs_dir is None:
            self.pdfs_dir = self.base_dir / os.getenv("PDFS_DIR", "pdfs")
        if self.extracted_dir is None:
            self.extracted_dir = self.base_dir / os.getenv("EXTRACTED_DIR", "extracted")
        if self.logs_dir is None:
            self.logs_dir = self.base_dir / os.getenv("LOG_DIR", "logs")
        
        # Ensure directories exist
        self.extracted_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def dump_raw_ocr(self) -> bool:
        """Whether to dump raw OCR output (enabled in debug mode)."""
        return self.debug or _get_bool_env("DUMP_RAW_OCR", False)
    
    @property
    def save_roi_overlays(self) -> bool:
        """Whether to save ROI overlay images (enabled in debug mode)."""
        return self.debug or _get_bool_env("SAVE_ROI_OVERLAYS", False)
    
    @property
    def keep_intermediate_files(self) -> bool:
        """Whether to keep intermediate processing files."""
        return self.debug or _get_bool_env("KEEP_INTERMEDIATE", False)
    
    def get_output_dir(self, pdf_name: str) -> Path:
        """Get output directory for a specific PDF."""
        return self.extracted_dir / pdf_name / "output"
    
    def get_images_dir(self, pdf_name: str) -> Path:
        """Get images directory for a specific PDF."""
        return self.extracted_dir / pdf_name / "images"
    
    def get_crops_dir(self, pdf_name: str) -> Path:
        """Get crops directory for a specific PDF."""
        return self.extracted_dir / pdf_name / "crops"


# Global config instance (lazily initialized)
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def reset_config() -> None:
    """Reset the global configuration (useful for testing)."""
    global _config
    _config = None
