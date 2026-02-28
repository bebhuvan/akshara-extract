"""
Project Akshara: Claude API Client
===================================

A robust Claude client with:
- PDF processing (up to 100 pages, 32MB)
- Image/vision support
- Prompt caching (5min default, 1hr optional)
- Automatic retry with backoff
- Cost tracking
- Updated Claude 4.6 model IDs

Claude is particularly good for:
- Assembly pass (better at following "don't edit" instructions)
- Validation pass (comparison tasks)
- Complex reasoning without "helpful editing"
"""

import os
import json
import time
import base64
from pathlib import Path
from typing import Optional, Dict, List, Any, Union
from dataclasses import dataclass

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


@dataclass
class TokenUsage:
    """Token usage tracking."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0

    def __add__(self, other: 'TokenUsage') -> 'TokenUsage':
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_creation_tokens=self.cache_creation_tokens + other.cache_creation_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
        )

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens + self.cache_creation_tokens + self.cache_read_tokens


@dataclass
class ClaudeResponse:
    """Structured response from Claude API."""
    text: str
    tokens: TokenUsage
    stop_reason: str
    duration_ms: int
    model: str
    success: bool
    error: Optional[str] = None
    json_data: Optional[Dict[str, Any]] = None


class ClaudeClient:
    """
    Full-featured Claude API client for Project Akshara.

    Features:
    - PDF processing (up to 100 pages)
    - Image/vision support
    - Prompt caching
    - Automatic retry

    Usage:
        client = ClaudeClient()

        # Text only
        response = client.generate("Analyze this text...")

        # With image
        response = client.generate_with_image(
            prompt="Transcribe this page...",
            image_path="/path/to/page.png"
        )

        # With PDF
        response = client.generate_with_pdf(
            prompt="Extract text from this document...",
            pdf_path="/path/to/book.pdf"
        )
    """

    # Model IDs — updated with Claude 4.6 models
    MODELS = {
        # Claude 4.6 (latest)
        "haiku-4.5": "claude-haiku-4-5-20251001",
        "sonnet-4.6": "claude-sonnet-4-6",
        "opus-4.6": "claude-opus-4-6",

        # Claude 4.5
        "sonnet-4.5": "claude-sonnet-4-5-20250929",
        "opus-4.5": "claude-opus-4-5-20251101",

        # Previous generation
        "haiku-3.5": "claude-3-5-haiku-20241022",
        "haiku-3": "claude-3-haiku-20240307",
        "sonnet-4": "claude-sonnet-4-20250514",
        "opus-4": "claude-opus-4-20250514",

        # Aliases — "haiku" points to 4.5 (best budget option)
        "haiku": "claude-haiku-4-5-20251001",
        "sonnet": "claude-sonnet-4-6",
        "opus": "claude-opus-4-6",
    }

    # Pricing per 1M tokens
    PRICING = {
        # Claude 4.6
        "claude-sonnet-4-6": {"input": 3.0, "output": 15.0, "cache_write": 3.75, "cache_read": 0.30},
        "claude-opus-4-6": {"input": 15.0, "output": 75.0, "cache_write": 18.75, "cache_read": 1.50},

        # Claude 4.5
        "claude-haiku-4-5-20251001": {"input": 1.0, "output": 5.0, "cache_write": 1.25, "cache_read": 0.10},
        "claude-sonnet-4-5-20250929": {"input": 3.0, "output": 15.0, "cache_write": 3.75, "cache_read": 0.30},
        "claude-opus-4-5-20251101": {"input": 5.0, "output": 25.0, "cache_write": 6.25, "cache_read": 0.50},

        # Previous generation
        "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.0, "cache_write": 1.0, "cache_read": 0.08},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25, "cache_write": 0.30, "cache_read": 0.03},
        "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0, "cache_write": 3.75, "cache_read": 0.30},
        "claude-opus-4-20250514": {"input": 15.0, "output": 75.0, "cache_write": 18.75, "cache_read": 1.50},
    }

    # Cache minimum tokens by model
    CACHE_MINIMUMS = {
        "claude-3-haiku-20240307": 2048,
        "claude-3-5-haiku-20241022": 2048,
        "claude-haiku-4-5-20251001": 4096,
        # All others: 1024
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "haiku",
        max_retries: int = 3,
        retry_delay: float = 2.0,
        timeout: int = 120,
        verbose: bool = True
    ):
        """
        Initialize Claude client.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            default_model: Default model ("haiku", "haiku-3.5", "haiku-4.5", "sonnet", "opus")
            max_retries: Maximum retry attempts
            retry_delay: Base delay between retries (exponential backoff)
            timeout: Request timeout in seconds
            verbose: Print status messages
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")

        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")

        self.default_model = self.MODELS.get(default_model, default_model)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.verbose = verbose

        # Initialize client
        self.client = anthropic.Anthropic(api_key=self.api_key)

        # Track total usage
        self.total_tokens = TokenUsage()
        self.total_cost = 0.0

    def _log(self, message: str):
        """Print if verbose mode enabled."""
        if self.verbose:
            print(f"[Claude] {message}")

    def _get_model_id(self, model: Optional[str]) -> str:
        """Resolve model alias to full ID."""
        model = model or self.default_model
        return self.MODELS.get(model, model)

    def _calculate_cost(self, tokens: TokenUsage, model: str) -> float:
        """Calculate cost for token usage."""
        pricing = self.PRICING.get(model, {"input": 0.25, "output": 1.25, "cache_write": 0.30, "cache_read": 0.03})

        input_cost = (tokens.input_tokens / 1_000_000) * pricing["input"]
        output_cost = (tokens.output_tokens / 1_000_000) * pricing["output"]
        cache_write_cost = (tokens.cache_creation_tokens / 1_000_000) * pricing["cache_write"]
        cache_read_cost = (tokens.cache_read_tokens / 1_000_000) * pricing["cache_read"]

        return input_cost + output_cost + cache_write_cost + cache_read_cost

    def _retry_with_backoff(self, func, *args, **kwargs) -> Any:
        """Execute function with exponential backoff retry."""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except anthropic.RateLimitError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    self._log(f"Rate limit hit, retry {attempt + 1}/{self.max_retries} in {delay:.1f}s")
                    time.sleep(delay)
            except anthropic.APIError as e:
                last_error = e
                error_str = str(e).lower()

                # Don't retry on certain errors
                if "invalid" in error_str and "key" in error_str:
                    raise

                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    self._log(f"API error, retry {attempt + 1}/{self.max_retries} in {delay:.1f}s: {e}")
                    time.sleep(delay)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    self._log(f"Error, retry {attempt + 1}/{self.max_retries} in {delay:.1f}s: {e}")
                    time.sleep(delay)

        raise last_error

    def _parse_response(self, response, model: str, duration_ms: int) -> ClaudeResponse:
        """Parse API response into structured format."""
        # Extract text
        text = ""
        if response.content:
            for block in response.content:
                if hasattr(block, 'text'):
                    text += block.text

        # Extract token usage
        usage = response.usage
        tokens = TokenUsage(
            input_tokens=getattr(usage, 'input_tokens', 0),
            output_tokens=getattr(usage, 'output_tokens', 0),
            cache_creation_tokens=getattr(usage, 'cache_creation_input_tokens', 0),
            cache_read_tokens=getattr(usage, 'cache_read_input_tokens', 0),
        )

        # Update totals
        self.total_tokens = self.total_tokens + tokens
        self.total_cost += self._calculate_cost(tokens, model)

        return ClaudeResponse(
            text=text,
            tokens=tokens,
            stop_reason=response.stop_reason,
            duration_ms=duration_ms,
            model=model,
            success=True
        )

    def _extract_json_object(self, text: str) -> Optional[Dict[str, Any]]:
        """Best-effort JSON object extraction for planner-style responses."""
        if not text:
            return None
        s = text.strip()
        if s.startswith("```"):
            s = s.removeprefix("```json").removeprefix("```").strip()
            if s.endswith("```"):
                s = s[:-3].strip()
        try:
            data = json.loads(s)
            return data if isinstance(data, dict) else None
        except Exception:
            pass
        start = s.find("{")
        if start < 0:
            return None
        depth = 0
        in_str = False
        esc = False
        for i, ch in enumerate(s[start:], start=start):
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        data = json.loads(s[start:i + 1])
                    except Exception:
                        return None
                    return data if isinstance(data, dict) else None
        return None

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        cache_system: bool = False,
        thinking: Optional[Any] = None,  # accepted for Gemini API parity; ignored
        json_mode: bool = False,
        json_schema: Optional[Dict[str, Any]] = None,  # accepted for parity; prompt still governs
    ) -> ClaudeResponse:
        """
        Generate text response.

        Args:
            prompt: The prompt text
            model: Model to use (defaults to default_model)
            system: Optional system instruction
            max_tokens: Maximum output tokens
            cache_system: Whether to cache the system prompt

        Returns:
            ClaudeResponse with generated text and metadata
        """
        model_id = self._get_model_id(model)
        start_time = time.time()

        try:
            # Build request
            kwargs = {
                "model": model_id,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }

            if system:
                if cache_system:
                    kwargs["system"] = [
                        {
                            "type": "text",
                            "text": system,
                            "cache_control": {"type": "ephemeral"}
                        }
                    ]
                else:
                    kwargs["system"] = system

            # Make request with retry
            response = self._retry_with_backoff(
                self.client.messages.create,
                **kwargs
            )

            duration_ms = int((time.time() - start_time) * 1000)
            parsed = self._parse_response(response, model_id, duration_ms)
            if json_mode or json_schema:
                parsed.json_data = self._extract_json_object(parsed.text)
            return parsed

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            return ClaudeResponse(
                text="",
                tokens=TokenUsage(),
                stop_reason="error",
                duration_ms=duration_ms,
                model=model_id,
                success=False,
                error=str(e)
            )

    def generate_with_image(
        self,
        prompt: str,
        image_path: Union[str, Path],
        model: Optional[str] = None,
        system: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> ClaudeResponse:
        """
        Generate response with image input.

        Args:
            prompt: The prompt text
            image_path: Path to image file (JPEG, PNG, GIF, WebP)
            model: Model to use
            system: Optional system instruction
            max_tokens: Maximum output tokens

        Returns:
            ClaudeResponse with generated text and metadata
        """
        model_id = self._get_model_id(model)
        image_path = Path(image_path)

        if not image_path.exists():
            return ClaudeResponse(
                text="",
                tokens=TokenUsage(),
                stop_reason="error",
                duration_ms=0,
                model=model_id,
                success=False,
                error=f"Image not found: {image_path}"
            )

        # Read and encode image
        with open(image_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")

        # Determine media type
        suffix = image_path.suffix.lower()
        media_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        media_type = media_types.get(suffix, "image/png")

        start_time = time.time()

        try:
            kwargs = {
                "model": model_id,
                "max_tokens": max_tokens,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_data,
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
            }

            if system:
                kwargs["system"] = system

            response = self._retry_with_backoff(
                self.client.messages.create,
                **kwargs
            )

            duration_ms = int((time.time() - start_time) * 1000)
            return self._parse_response(response, model_id, duration_ms)

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            return ClaudeResponse(
                text="",
                tokens=TokenUsage(),
                stop_reason="error",
                duration_ms=duration_ms,
                model=model_id,
                success=False,
                error=str(e)
            )

    def generate_with_pdf(
        self,
        prompt: str,
        pdf_path: Union[str, Path],
        model: Optional[str] = None,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        cache_pdf: bool = False,
    ) -> ClaudeResponse:
        """
        Generate response with PDF input.

        Claude PDF support:
        - Max 100 pages per request
        - Max 32MB request size
        - Converts pages to images + extracts text
        - ~1,500-3,000 tokens per page

        Args:
            prompt: The prompt text
            pdf_path: Path to PDF file
            model: Model to use
            system: Optional system instruction
            max_tokens: Maximum output tokens
            cache_pdf: Whether to cache the PDF (for repeated queries)

        Returns:
            ClaudeResponse with generated text and metadata
        """
        model_id = self._get_model_id(model)
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            return ClaudeResponse(
                text="",
                tokens=TokenUsage(),
                stop_reason="error",
                duration_ms=0,
                model=model_id,
                success=False,
                error=f"PDF not found: {pdf_path}"
            )

        # Check file size (32MB limit)
        file_size = pdf_path.stat().st_size
        if file_size > 32 * 1024 * 1024:
            return ClaudeResponse(
                text="",
                tokens=TokenUsage(),
                stop_reason="error",
                duration_ms=0,
                model=model_id,
                success=False,
                error=f"PDF exceeds 32MB limit: {file_size / (1024*1024):.1f}MB"
            )

        # Read and encode PDF
        with open(pdf_path, "rb") as f:
            pdf_data = base64.standard_b64encode(f.read()).decode("utf-8")

        start_time = time.time()

        try:
            # Build document content
            document_block = {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": pdf_data
                }
            }

            if cache_pdf:
                document_block["cache_control"] = {"type": "ephemeral"}

            kwargs = {
                "model": model_id,
                "max_tokens": max_tokens,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            document_block,
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
            }

            if system:
                kwargs["system"] = system

            response = self._retry_with_backoff(
                self.client.messages.create,
                **kwargs
            )

            duration_ms = int((time.time() - start_time) * 1000)
            return self._parse_response(response, model_id, duration_ms)

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            return ClaudeResponse(
                text="",
                tokens=TokenUsage(),
                stop_reason="error",
                duration_ms=duration_ms,
                model=model_id,
                success=False,
                error=str(e)
            )

    def get_usage_summary(self) -> Dict[str, Any]:
        """Get summary of API usage."""
        return {
            "total_tokens": {
                "input": self.total_tokens.input_tokens,
                "output": self.total_tokens.output_tokens,
                "cache_creation": self.total_tokens.cache_creation_tokens,
                "cache_read": self.total_tokens.cache_read_tokens,
                "total": self.total_tokens.total,
            },
            "total_cost_usd": self.total_cost,
        }

    def reset_usage(self):
        """Reset usage tracking."""
        self.total_tokens = TokenUsage()
        self.total_cost = 0.0


# =============================================================================
# ASSEMBLY PROMPTS FOR CLAUDE
# =============================================================================

ASSEMBLY_SYSTEM_PROMPT = """You are a text assembler for Project Akshara, an archival digitization project.

CRITICAL RULES:
1. You MUST NOT edit, correct, or improve any text content.
2. You MUST preserve all archaic spellings, grammar, and formatting.
3. You are only allowed to:
   - Join words split across pages (e.g., "archae-" + "ology" = "archaeology")
   - Merge paragraphs that were split by page breaks
   - Place footnotes at the end of chapters
4. If you are uncertain, DO NOT change anything.
5. Output the assembled text exactly as it should appear.

You are a mechanical assembler, not an editor."""


JOIN_WORDS_PROMPT = """Join these text fragments from adjacent pages.

END OF PAGE {prev_page}:
---
{prev_tail}
---

START OF PAGE {next_page}:
---
{next_head}
---

TASK:
1. If the previous page ends with a hyphenated word fragment, join it with the continuation on the next page.
2. If the previous page ends mid-sentence (no terminal punctuation), the next page continues it.
3. Output ONLY the corrected junction (last ~150 chars of prev + first ~150 chars of next).
4. Do NOT edit, correct, or improve any text.

OUTPUT:
---JUNCTION---
[corrected text around the page break]
---END---"""


FOOTNOTE_PLACEMENT_PROMPT = """Place these footnotes at the end of the chapter.

CHAPTER TEXT:
---
{chapter_text}
---

COLLECTED FOOTNOTES:
---
{footnotes}
---

TASK:
1. Verify each footnote marker in the text has a corresponding footnote.
2. Place all footnotes at the END of the chapter using [^N]: format.
3. Do NOT modify any text content.
4. Report any orphaned markers or footnotes.

OUTPUT:
---ASSEMBLED---
[chapter text unchanged]

---

### Notes

[^1]: [footnote 1 text]
[^2]: [footnote 2 text]
---END---

ISSUES (if any):
- [list any marker/footnote mismatches]"""


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_client(**kwargs) -> ClaudeClient:
    """Create a new Claude client with default settings."""
    return ClaudeClient(**kwargs)


def load_env():
    """Load environment variables from .env file."""
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                key, _, value = line.partition("=")
                os.environ[key.strip()] = value.strip()


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    load_env()

    print("Testing Claude Client...")
    print()

    try:
        client = create_client(verbose=True)

        # Test text generation
        print("1. Testing text generation...")
        response = client.generate(
            prompt="Say 'Hello, Akshara!' and nothing else.",
            system="You are a helpful assistant. Be concise.",
        )

        print(f"   Response: {response.text}")
        print(f"   Success: {response.success}")
        print(f"   Tokens: {response.tokens.total}")
        print(f"   Duration: {response.duration_ms}ms")
        print()

        # Print usage summary
        summary = client.get_usage_summary()
        print(f"2. Usage Summary:")
        print(f"   Total tokens: {summary['total_tokens']['total']}")
        print(f"   Total cost: ${summary['total_cost_usd']:.6f}")

    except ImportError:
        print("anthropic package not installed. Run: pip install anthropic")
    except Exception as e:
        print(f"Error: {e}")
