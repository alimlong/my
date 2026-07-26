from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional

import cloudscraper
import requests


ChunkType = Literal["reasoning", "content", "raw", "done", "error"]


@dataclass
class ChunkEvent:
    """One parsed stream fragment."""

    type: ChunkType
    text: str
    index: int
    elapsed_s: float
    raw: Optional[Dict[str, Any]] = None


@dataclass
class StreamMetrics:
    started_at: float = field(default_factory=time.perf_counter)
    first_chunk_at: Optional[float] = None
    first_content_at: Optional[float] = None
    last_content_at: Optional[float] = None
    content_tokens: int = 0
    reasoning_tokens: int = 0
    chunks: int = 0

    @property
    def ttfc_s(self) -> Optional[float]:
        """Time to first streamed chunk."""
        return None if self.first_chunk_at is None else self.first_chunk_at - self.started_at

    @property
    def ttft_s(self) -> Optional[float]:
        """Time to first emitted text fragment."""
        return None if self.first_content_at is None else self.first_content_at - self.started_at

    @property
    def tpot_s(self) -> Optional[float]:
        """
        Time per output token.

        By default this parser treats each emitted text fragment as one token,
        because OpenAI-style streaming chunks do not carry token counts. Pass a
        tokenizer/count function to StreamChunkParser to calculate real token
        counts.
        """
        total_tokens = self.content_tokens + self.reasoning_tokens
        if (
            total_tokens <= 1
            or self.first_content_at is None
            or self.last_content_at is None
        ):
            return None
        return (self.last_content_at - self.first_content_at) / (total_tokens - 1)


class ThinkTagSplitter:
    """
    Incrementally split content that may contain <think>...</think>.

    The buffer is kept long enough to detect tags even if a tag is split across
    multiple network chunks.
    """

    START = "<think>"
    END = "</think>"
    MAX_TAG_LEN = max(len(START), len(END))

    def __init__(self) -> None:
        self._buffer = ""
        self._in_think = False

    def feed(self, text: str, final: bool = False) -> List[tuple[ChunkType, str]]:
        self._buffer += text
        events: List[tuple[ChunkType, str]] = []

        while self._buffer:
            if self._in_think:
                tag = self._buffer.find(self.END)
                if tag >= 0:
                    if tag:
                        events.append(("reasoning", self._buffer[:tag]))
                    self._buffer = self._buffer[tag + len(self.END) :]
                    self._in_think = False
                    continue

                emit_len = len(self._buffer) if final else self._safe_emit_len(self._buffer)
                if emit_len <= 0:
                    break
                events.append(("reasoning", self._buffer[:emit_len]))
                self._buffer = self._buffer[emit_len:]
                continue

            tag = self._buffer.find(self.START)
            if tag >= 0:
                if tag:
                    events.append(("content", self._buffer[:tag]))
                self._buffer = self._buffer[tag + len(self.START) :]
                self._in_think = True
                continue

            emit_len = len(self._buffer) if final else self._safe_emit_len(self._buffer)
            if emit_len <= 0:
                break
            events.append(("content", self._buffer[:emit_len]))
            self._buffer = self._buffer[emit_len:]

        return [(kind, value) for kind, value in events if value]

    def flush(self) -> List[tuple[ChunkType, str]]:
        return self.feed("", final=True)

    @classmethod
    def _safe_emit_len(cls, buffer: str) -> int:
        keep = cls.MAX_TAG_LEN - 1
        if len(buffer) <= keep:
            return 0
        return len(buffer) - keep


class StreamChunkParser:
    """
    Parse OpenAI-compatible streaming responses.

    Supports both formats:
    1. delta.content + delta.reasoning_content/reasoning fields.
    2. all text in delta.content with <think>...</think> tags.
    """

    REASONING_KEYS = (
        "reasoning_content",
        "reasoning",
        "reasoningContent",
        "reasioning_content",
    )

    def __init__(
        self,
        token_counter: Optional[Callable[[str], int]] = None,
        split_think_tags: bool = True,
    ) -> None:
        self.metrics = StreamMetrics()
        self._index = 0
        self._splitter = ThinkTagSplitter()
        self._token_counter = token_counter or (lambda text: 1 if text else 0)
        self._split_think_tags = split_think_tags

    def parse_lines(self, lines: Iterator[str]) -> Iterator[ChunkEvent]:
        for line in lines:
            # print("line:", line)
            payload = self._extract_sse_payload(line)
            if payload is None:
                continue

            if payload == "[DONE]":
                yield from self._flush()
                yield self._event("done", "")
                return

            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                yield self._event("raw", payload)
                continue

            yield from self.parse_json_chunk(chunk)

        yield from self._flush()

    def parse_json_chunk(self, chunk: Dict[str, Any]) -> Iterator[ChunkEvent]:
        choice = (chunk.get("choices") or [{}])[0]
        delta = choice.get("delta") or choice.get("message") or {}

        for key in self.REASONING_KEYS:
            reasoning_text = delta.get(key)
            if reasoning_text:
                yield self._event("reasoning", str(reasoning_text), raw=chunk)

        content = delta.get("content")
        if content:
            if self._split_think_tags:
                for kind, text in self._splitter.feed(str(content)):
                    yield self._event(kind, text, raw=chunk)
            else:
                yield self._event("content", str(content), raw=chunk)

        if choice.get("finish_reason"):
            yield from self._flush()

    def _flush(self) -> Iterator[ChunkEvent]:
        for kind, text in self._splitter.flush():
            yield self._event(kind, text)

    def _event(
        self,
        kind: ChunkType,
        text: str,
        raw: Optional[Dict[str, Any]] = None,
    ) -> ChunkEvent:
        now = time.perf_counter()
        self.metrics.chunks += 1
        if self.metrics.first_chunk_at is None:
            self.metrics.first_chunk_at = now

        if kind in ("reasoning", "content") and text:
            if self.metrics.first_content_at is None:
                self.metrics.first_content_at = now
            self.metrics.last_content_at = now
            tokens = self._token_counter(text)
            if kind == "reasoning":
                self.metrics.reasoning_tokens += tokens
            else:
                self.metrics.content_tokens += tokens

        event = ChunkEvent(kind, text, self._index, now - self.metrics.started_at, raw)
        self._index += 1
        return event

    @staticmethod
    def _extract_sse_payload(line: str) -> Optional[str]:
        line = line.strip()
        if not line:
            return None
        if line.startswith("data:"):
            return line[5:].strip()
        return line


def stream_chat_completion(
    url: str,
    payload: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 300,
    token_counter: Optional[Callable[[str], int]] = None,
) -> Iterator[ChunkEvent]:
    """
    Send a streaming request with requests and yield parsed ChunkEvent objects.
    """

    body = dict(payload)
    body["stream"] = True
    parser = StreamChunkParser(token_counter=token_counter)

    scraper = cloudscraper.create_scraper()

    with requests.post(
        url,
        headers=headers,
        json=body,
        stream=True,
        timeout=timeout,
    ) as response:
        if response.status_code != 200:
            print(response.text)
        response.raise_for_status()
        for k in response.request.headers:
            print(k, ':', response.request.headers.get(k))
        print('#################################')
        for k in response.headers:
            print(k, ':', response.headers.get(k))
        print(response.encoding, response.apparent_encoding)
        response.encoding = 'utf-8'
        lines = response.iter_lines(decode_unicode=True)
        yield from parser.parse_lines(lines)

        metrics = parser.metrics
        print(
            {
                "ttfc_s": metrics.ttfc_s,
                "ttft_s": metrics.ttft_s,
                "tpot_s": metrics.tpot_s,
                "content_tokens": metrics.content_tokens,
                "reasoning_tokens": metrics.reasoning_tokens,
                "chunks": metrics.chunks,
            }
        )


def models(url, api_key):
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.get(url, headers=headers)
    if resp.status_code == 200:
        print(json.dumps(resp.json(), indent=2, ensure_ascii=False))
    else:
        print(resp.text)


if __name__ == "__main__":
    # Example:
    # python stream_chunk_parser.py
    api_url = "https://www.ccgpai.com/v1/chat/completions"
    # api_url = "https://www.ccgpai.com/v1/responses"
    api_key = ""
    req = {
        "model": "gpt-5.5",
        "messages": [{"role": "user", "content": "9.11和9.9哪个大？"}],
        "temperature": 0,
        "stream": True
    }

    # req = {
    #     'model': 'gpt-5.5',
    #     'input': '你好，你是什么模型'
    # }

    # models('https://www.ccgpai.com/v1/models', api_key)

    for item in stream_chat_completion(
        api_url,
        req,
        headers={
            "Authorization": f"Bearer {api_key}",
            # "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Accept": "text/event-stream,text/html,application/json",
            "HOST": "www.ccgpai.com",
            "Origin": "https://www.ccgpai.com",
            # "Referer": "https://www.ccgpai.com/",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36"
            },
    ):
        print(f"{item.index:04d} {item.type:9s} {item.elapsed_s:.3f}s {item.text!r}")
