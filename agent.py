import asyncio
import json
import logging
import re
import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

try:
    from .agent_loop import run_agent
except ImportError:
    from agent_loop import run_agent

logger = logging.getLogger(__name__)

app = FastAPI()
PING_INTERVAL_SECONDS = 5
STREAM_CHUNK_CHARS = 24


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    question: str
    chat_history: Optional[list] = None


class QueryResponse(BaseModel):
    answer: str


def _sse_event(event: str, data: dict | None = None) -> str:
    lines = [f"event: {event}"]
    if data is not None:
        lines.append(f"data: {json.dumps(data, ensure_ascii=False)}")
    return "\n".join(lines) + "\n\n"


def _iter_answer_chunks(answer: str):
    text = (answer or "").strip()
    if not text:
        yield "未能找到答案。"
        return

    word_chunks = [chunk for chunk in re.findall(r"\S+|\s+\S+", text) if chunk]
    if len(word_chunks) > 1:
        for chunk in word_chunks:
            yield chunk
        return

    for idx in range(0, len(text), STREAM_CHUNK_CHARS):
        yield text[idx: idx + STREAM_CHUNK_CHARS]


def _run_agent_sync(question: str) -> str:
    """Run the async agent in a dedicated event loop (for use with to_thread)."""
    return asyncio.run(run_agent(question))


@app.post("/")
async def query(req: QueryRequest):
    async def stream_response():
        question = (req.question or "").strip()
        if not question:
            yield _sse_event("Message", {"answer": "无法识别问题，请重新输入。"})
            return

        task = asyncio.create_task(asyncio.to_thread(_run_agent_sync, question))
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=PING_INTERVAL_SECONDS)
                if task in done:
                    break
                yield _sse_event("Ping")

            try:
                answer = task.result()
            except Exception as exc:
                logger.error(f"Agent error: {exc}", exc_info=True)
                answer = f"处理出错: {exc}"

            for chunk in _iter_answer_chunks(answer):
                yield _sse_event("Message", {"answer": chunk})
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# -----------  AG-UI Protocol  -----------

def _extract_question_from_agui(data: dict) -> str:
    """从 AG-UI 请求中提取用户最后一条消息作为 question"""
    messages = data.get("messages", [])
    for msg in reversed(messages):
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user" and content:
            return content
    return ""


def _agui_event(event_type: str, **kwargs) -> str:
    """构造一条 AG-UI SSE data 行"""
    payload: Dict[str, Any] = {"type": event_type}
    payload.update({k: v for k, v in kwargs.items() if v is not None})
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/ag-ui")
async def ag_ui(request: Request) -> StreamingResponse:
    data = await request.json()

    thread_id = data.get("threadId", "")
    run_id = data.get("runId", "")
    question = _extract_question_from_agui(data)

    async def stream_response():
        msg_id = str(uuid.uuid4())

        # RUN_STARTED
        yield _agui_event("RUN_STARTED", threadId=thread_id, runId=run_id)

        if not question:
            # 没有有效问题，直接结束
            yield _agui_event("TEXT_MESSAGE_START", messageId=msg_id, role="assistant")
            yield _agui_event("TEXT_MESSAGE_CONTENT", messageId=msg_id, delta="无法识别问题，请重新输入。")
            yield _agui_event("TEXT_MESSAGE_END", messageId=msg_id)
            yield _agui_event("RUN_FINISHED", threadId=thread_id, runId=run_id)
            return

        task = asyncio.create_task(asyncio.to_thread(_run_agent_sync, question))
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=PING_INTERVAL_SECONDS)
                if task in done:
                    break
                yield ": keepalive\n\n"

            try:
                answer = task.result()
            except Exception as e:
                logger.error(f"Agent error: {e}", exc_info=True)
                answer = f"处理出错: {e}"
        finally:
            if not task.done():
                task.cancel()

        if not answer:
            answer = "未能找到答案。"

        # TEXT_MESSAGE_START -> CONTENT -> END
        yield _agui_event("TEXT_MESSAGE_START", messageId=msg_id, role="assistant")
        yield _agui_event("TEXT_MESSAGE_CONTENT", messageId=msg_id, delta=answer)
        yield _agui_event("TEXT_MESSAGE_END", messageId=msg_id)

        # RUN_FINISHED
        yield _agui_event("RUN_FINISHED", threadId=thread_id, runId=run_id)

    return StreamingResponse(stream_response(), media_type="text/event-stream")
