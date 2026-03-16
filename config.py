"""Runtime constants from environment variables + LLM call helper."""

import logging
import os

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# ── LLM ──────────────────────────────────────────────────────────────
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3-max")
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")

# ── Search ───────────────────────────────────────────────────────────
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")

# ── Debug ────────────────────────────────────────────────────────────
DEBUG = os.getenv("DEBUG", "0") == "1"
NO_TIMEOUT = os.getenv("NO_TIMEOUT", "0") == "1"

# ── Resource controls ────────────────────────────────────────────────
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "30"))
MAX_SEARCH_QUERIES = int(os.getenv("MAX_SEARCH_QUERIES", "25"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "180"))           # default LLM call timeout (seconds)
MAX_TOTAL_SECONDS = 999999 if NO_TIMEOUT else int(os.getenv("TOTAL_TIMEOUT", "600"))
MAX_RESULTS_PER_QUERY = int(os.getenv("MAX_RESULTS_PER_QUERY", "8"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2000"))  # max output tokens per LLM call

# ── Page fetch controls ─────────────────────────────────────────────
MAX_FETCH_PAGES = int(os.getenv("MAX_FETCH_PAGES", "6"))    # max pages fetched per question
FETCH_TIMEOUT = int(os.getenv("FETCH_TIMEOUT", "15"))       # single page timeout (seconds)
MAX_PAGE_CHARS = int(os.getenv("MAX_PAGE_CHARS", "15000"))   # max chars sent to LLM per page

# ── LLM call via requests ────────────────────────────────────────────
# Cloudflare blocks the default openai-python User-Agent on this endpoint,
# so we bypass the SDK entirely and use requests with a browser-like UA.
import time as _time  # noqa: E402
import requests as _requests  # noqa: E402

_LLM_ENDPOINT = f"{BASE_URL}/chat/completions"
_LLM_HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}

_LLM_MAX_RETRIES = 3
_LLM_BACKOFF = [3, 5, 10]


def call_llm(prompt: str, temperature: float = 0.1, timeout: int = LLM_TIMEOUT) -> str:
    """Synchronous LLM call using requests with retry. Returns content string."""
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": LLM_MAX_TOKENS,
        "enable_thinking": False,  # Disable thinking mode for speed
        "stop": ["\nObservation:", "\nObservation", "\nResult:", "\nResult"],
    }

    last_err = None
    for attempt in range(_LLM_MAX_RETRIES):
        try:
            resp = _requests.post(
                _LLM_ENDPOINT,
                json=payload,
                headers=_LLM_HEADERS,
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()

            content = data["choices"][0]["message"]["content"]

            # Strip <think>...</think> tags in case they still appear
            if "</think>" in content:
                idx = content.index("</think>")
                content = content[idx + len("</think>"):]

            return content.strip()

        except Exception as e:
            last_err = e
            if attempt < _LLM_MAX_RETRIES - 1:
                _time.sleep(_LLM_BACKOFF[attempt])

    raise last_err  # type: ignore


# ── Logging setup ──────────────────────────────────────────────────
def setup_logging(log_file: str | None = None):
    """统一配置日志系统。

    - DEBUG=0: 控制台只输出 WARNING+，无文件输出
    - DEBUG=1: 控制台输出 INFO+，文件输出 DEBUG+（全量）
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if DEBUG else logging.WARNING)

    # 清理已有 handlers（避免重复配置）
    root.handlers.clear()

    # 格式
    fmt_console = logging.Formatter("[%(levelname).1s] %(message)s")
    fmt_file = logging.Formatter(
        "%(asctime)s [%(levelname).1s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # 控制台 handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO if DEBUG else logging.WARNING)
    console.setFormatter(fmt_console)
    root.addHandler(console)

    # 文件 handler（仅 DEBUG 模式且指定了路径）
    if DEBUG and log_file:
        fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt_file)
        root.addHandler(fh)
