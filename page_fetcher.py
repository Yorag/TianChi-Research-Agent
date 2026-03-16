"""Page fetcher: Jina Reader API (primary) + direct fetch fallback.

Notes:
- Some environments have a brotli/aiohttp incompatibility that breaks decoding
  of `Content-Encoding: br`. To keep fetch robust, we explicitly request only
  `gzip, deflate`.
- PDFs are common in search results (theses, reports). When possible, extract
  text from PDFs for the agent.
"""

import re
import logging
from io import BytesIO

import aiohttp

try:
    from .config import FETCH_TIMEOUT, MAX_PAGE_CHARS
except ImportError:
    from config import FETCH_TIMEOUT, MAX_PAGE_CHARS

logger = logging.getLogger(__name__)

_JINA_PREFIX = "https://r.jina.ai/"

_BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

# Avoid brotli decoding issues by not requesting br.
_ACCEPT_ENCODING = "gzip, deflate"

try:
    from pypdf import PdfReader  # type: ignore
except Exception:  # pragma: no cover
    PdfReader = None


def _strip_html(html: str) -> str:
    """Remove script/style blocks and HTML tags, return plain text."""
    text = re.sub(r"<script[^>]*>[\s\S]*?</script>", "", html, flags=re.IGNORECASE)
    text = re.sub(r"<style[^>]*>[\s\S]*?</style>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


async def fetch_page_content(url: str) -> str:
    """Fetch a web page and return its text content (max MAX_PAGE_CHARS chars).

    1. Try Jina Reader API (returns clean Markdown).
    2. Fallback: direct aiohttp GET + HTML stripping.
    Returns empty string on total failure.
    """
    is_pdf_url = url.lower().split("?")[0].endswith(".pdf")

    # --- Jina Reader (best effort for HTML) ---
    if not is_pdf_url:
        try:
            jina_url = f"{_JINA_PREFIX}{url}"
            timeout = aiohttp.ClientTimeout(total=FETCH_TIMEOUT)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    jina_url,
                    headers={
                        "User-Agent": _BROWSER_UA,
                        "Accept-Encoding": _ACCEPT_ENCODING,
                    },
                ) as resp:
                    if resp.status == 200:
                        text = await resp.text(errors="replace")
                        text = text.strip()
                        if text and len(text) > 100:
                            logger.debug(f"fetch(jina): got {len(text)} chars from {url[:60]}")
                            return text[:MAX_PAGE_CHARS]
        except Exception as e:
            logger.debug(f"fetch(jina): failed for {url[:60]}: {e}")

    # --- Fallback: direct fetch (HTML/PDF) ---
    try:
        timeout = aiohttp.ClientTimeout(total=FETCH_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(
                url,
                headers={
                    "User-Agent": _BROWSER_UA,
                    "Accept-Encoding": _ACCEPT_ENCODING,
                },
                allow_redirects=True,
            ) as resp:
                if resp.status == 200:
                    content_type = (resp.headers.get("Content-Type") or "").lower()
                    is_pdf = is_pdf_url or ("application/pdf" in content_type) or ("pdf" in content_type)

                    if is_pdf:
                        if PdfReader is None:
                            logger.debug("fetch(direct): pypdf not installed; cannot parse PDF")
                            return ""
                        data = await resp.read()
                        reader = PdfReader(BytesIO(data))
                        chunks: list[str] = []
                        total_chars = 0
                        for page in reader.pages:
                            page_text = page.extract_text() or ""
                            if not page_text:
                                continue
                            chunks.append(page_text)
                            total_chars += len(page_text)
                            if total_chars >= MAX_PAGE_CHARS:
                                break
                        text = "\n".join(chunks).strip()
                        if text and len(text) > 100:
                            logger.debug(f"fetch(direct/pdf): got {len(text)} chars from {url[:60]}")
                            return text[:MAX_PAGE_CHARS]
                        return ""

                    html = await resp.text(errors="replace")
                    text = _strip_html(html)
                    if text and len(text) > 100:
                        logger.debug(f"fetch(direct): got {len(text)} chars from {url[:60]}")
                        return text[:MAX_PAGE_CHARS]
    except Exception as e:
        logger.debug(f"fetch(direct): failed for {url[:60]}: {e}")

    return ""
