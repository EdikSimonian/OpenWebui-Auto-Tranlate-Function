"""
title: OpenWebUI Auto Translate
author: Edik Simonian
author_url: https://github.com/open-webui
version: 1.0.0
description: Auto-detects non-English user input, translates it to English for the LLM, then translates the response back to the user's original language. Uses Google Translate (free, no API key) with MyMemory as fallback.
required_open_webui_version: 0.4.0
requirements: langdetect, deep-translator
"""

import asyncio
import hashlib
import re
from collections import OrderedDict
from typing import Optional, Callable, Dict, Tuple, Any, List

from pydantic import BaseModel, Field

try:
    from langdetect import detect_langs, DetectorFactory

    DetectorFactory.seed = 0
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

try:
    from deep_translator import GoogleTranslator, MyMemoryTranslator

    TRANSLATOR_AVAILABLE = True
except ImportError:
    TRANSLATOR_AVAILABLE = False

LANGUAGE_MAP = {
    "af": "Afrikaans",
    "sq": "Albanian",
    "am": "Amharic",
    "ar": "Arabic",
    "hy": "Armenian",
    "az": "Azerbaijani",
    "eu": "Basque",
    "be": "Belarusian",
    "bn": "Bengali",
    "bs": "Bosnian",
    "bg": "Bulgarian",
    "ca": "Catalan",
    "zh-cn": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)",
    "hr": "Croatian",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "et": "Estonian",
    "fi": "Finnish",
    "fr": "French",
    "gl": "Galician",
    "ka": "Georgian",
    "de": "German",
    "el": "Greek",
    "gu": "Gujarati",
    "he": "Hebrew",
    "hi": "Hindi",
    "hu": "Hungarian",
    "is": "Icelandic",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "kn": "Kannada",
    "kk": "Kazakh",
    "ko": "Korean",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "mk": "Macedonian",
    "ms": "Malay",
    "ml": "Malayalam",
    "mr": "Marathi",
    "mn": "Mongolian",
    "ne": "Nepali",
    "no": "Norwegian",
    "fa": "Persian",
    "pl": "Polish",
    "pt": "Portuguese",
    "pa": "Punjabi",
    "ro": "Romanian",
    "ru": "Russian",
    "sr": "Serbian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "es": "Spanish",
    "sw": "Swahili",
    "sv": "Swedish",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "uz": "Uzbek",
    "vi": "Vietnamese",
    "cy": "Welsh",
}

# langdetect → deep-translator code mapping for codes that differ
_CODE_MAP = {
    "zh-cn": "zh-CN",
    "zh-tw": "zh-TW",
}

_MAX_CHAT_LANG_ENTRIES = 1000


class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=0,
            description="Filter execution priority. Lower values run first.",
        )
        detection_threshold: float = Field(
            default=0.75,
            description="Minimum confidence for language detection (0.0 to 1.0).",
        )
        show_status: bool = Field(
            default=True,
            description="Show translation status messages in the chat UI.",
        )

    class UserValves(BaseModel):
        enabled: bool = Field(
            default=True,
            description="Enable automatic translation for your messages.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self._chat_languages: OrderedDict[str, str] = OrderedDict()
        self._translation_cache: Dict[str, str] = {}
        self._max_cache_size = 500

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_user_enabled(user_valves: Any) -> bool:
        if user_valves is None:
            return True
        return getattr(user_valves, "enabled", True)

    def _resolve_chat_id(
        self,
        __metadata__: Optional[dict],
        __user__: Optional[dict],
        body: dict,
    ) -> str:
        chat_id = (__metadata__ or {}).get("chat_id") or body.get("chat_id")
        if chat_id:
            return str(chat_id)
        user_id = (__user__ or {}).get("id", "unknown")
        return f"_no_chat_{user_id}"

    def _store_chat_language(self, chat_id: str, lang: str) -> None:
        if chat_id in self._chat_languages:
            self._chat_languages.move_to_end(chat_id)
        self._chat_languages[chat_id] = lang
        while len(self._chat_languages) > _MAX_CHAT_LANG_ENTRIES:
            self._chat_languages.popitem(last=False)

    async def _emit_status(
        self, emitter: Optional[Callable], description: str, done: bool
    ) -> None:
        if not emitter or not self.valves.show_status:
            return
        try:
            await emitter(
                {"type": "status", "data": {"description": description, "done": done}}
            )
        except Exception:
            pass

    @staticmethod
    def _translator_code(lang: str) -> str:
        """Convert a langdetect code to a deep-translator compatible code."""
        return _CODE_MAP.get(lang, lang)

    # ------------------------------------------------------------------
    # Language detection
    # ------------------------------------------------------------------

    def _detect_language(self, text: str) -> Tuple[str, float]:
        clean = text.strip()
        if not clean:
            return "en", 1.0

        # Script detection is checked first for non-Latin scripts.
        # These are unambiguous — if text is mostly Armenian characters it IS
        # Armenian, regardless of what langdetect thinks.
        script_lang, script_conf = self._detect_by_script(clean)
        if script_lang != "en" and script_conf >= 0.6:
            return script_lang, script_conf

        # For Latin-script languages (French, Spanish, German…) use langdetect.
        if LANGDETECT_AVAILABLE and len(clean) >= 8:
            try:
                results = detect_langs(clean)
                if results:
                    return str(results[0].lang), results[0].prob
            except Exception:
                pass

        return script_lang, script_conf

    @staticmethod
    def _detect_by_script(text: str) -> Tuple[str, float]:
        script_counts: Dict[str, int] = {}
        alpha_total = 0

        for char in text:
            cp = ord(char)
            lang = None
            # Armenian (U+0530–U+058F) and Armenian ligatures (U+FB13–U+FB17)
            if 0x0530 <= cp <= 0x058F or 0xFB13 <= cp <= 0xFB17:
                lang = "hy"
            elif 0x0400 <= cp <= 0x04FF:
                lang = "ru"
            elif 0x0600 <= cp <= 0x06FF:
                lang = "ar"
            elif 0x4E00 <= cp <= 0x9FFF:
                lang = "zh-cn"
            elif 0x3040 <= cp <= 0x309F or 0x30A0 <= cp <= 0x30FF:
                lang = "ja"
            elif 0xAC00 <= cp <= 0xD7AF:
                lang = "ko"
            elif 0x0900 <= cp <= 0x097F:
                lang = "hi"
            elif 0x0E00 <= cp <= 0x0E7F:
                lang = "th"
            elif 0x10A0 <= cp <= 0x10FF:
                lang = "ka"  # Georgian
            elif 0x1200 <= cp <= 0x137F:
                lang = "am"  # Amharic / Ethiopic

            if lang:
                script_counts[lang] = script_counts.get(lang, 0) + 1
                alpha_total += 1
            elif char.isalpha():
                alpha_total += 1

        if not script_counts or alpha_total == 0:
            return "en", 0.5

        dominant = max(script_counts, key=script_counts.get)  # type: ignore[arg-type]
        ratio = script_counts[dominant] / alpha_total
        if ratio >= 0.3:
            return dominant, min(0.6 + ratio * 0.3, 0.95)
        return "en", 0.5

    # ------------------------------------------------------------------
    # Code-block protection
    # ------------------------------------------------------------------

    # Capturing group so re.split() keeps the matched code blocks in the
    # result list.  Fenced blocks are tried before inline spans so that
    # triple-backtick fences always win over single-backtick matches.
    _CODE_FENCE_RE = re.compile(
        r"(```[\s\S]*?```|~~~[\s\S]*?~~~|`[^`\n]+`)",
        re.MULTILINE,
    )

    # ------------------------------------------------------------------
    # Translation via Google Translate / MyMemory (no API key)
    # ------------------------------------------------------------------

    async def _translate_text(
        self, text: str, source_lang: str, target_lang: str
    ) -> str:
        """Translate *text* while keeping every code block intact.

        Strategy: split the text on code spans/blocks using a capturing-group
        re.split().  This gives alternating segments:
            [prose, code, prose, code, …]
        Only the prose parts (even indices) are sent to the translator.
        Code parts (odd indices) are passed through unchanged.  The segments
        are then reassembled in order.
        """
        if source_lang == target_lang or not text.strip():
            return text

        if not TRANSLATOR_AVAILABLE:
            print("[AutoTranslate] deep-translator not installed — skipping translation.")
            return text

        # Full-text cache (keyed on the original text before splitting).
        cache_key = hashlib.sha256(
            f"{source_lang}|{target_lang}|{text}".encode()
        ).hexdigest()
        if cache_key in self._translation_cache:
            return self._translation_cache[cache_key]

        segments = self._CODE_FENCE_RE.split(text)
        # segments = [prose0, code0, prose1, code1, …]
        # Even indices → prose to translate; odd indices → code to keep.

        translated_segments: List[str] = []
        for i, segment in enumerate(segments):
            if i % 2 == 1:
                # Code block — never touch it.
                translated_segments.append(segment)
            elif segment.strip():
                # Prose — strip surrounding newlines, translate, restore them.
                # Google Translate drops leading/trailing whitespace, which
                # would cause the adjacent fenced block to lose its own line
                # and render as broken inline red text instead of a code window.
                leading_nl = segment[: len(segment) - len(segment.lstrip("\n"))]
                trailing_nl = segment[len(segment.rstrip("\n")) :]
                prose = segment.strip("\n")
                translated = await self._translate_segment(prose, source_lang, target_lang)
                translated_segments.append(leading_nl + translated + trailing_nl)
            else:
                # Newline-only gap — keep as-is (preserves blank lines, etc.)
                translated_segments.append(segment)

        result = "".join(translated_segments)
        self._cache_put(cache_key, result)
        return result

    async def _translate_segment(
        self, text: str, source_lang: str, target_lang: str
    ) -> str:
        """Send a single prose segment (guaranteed code-free) to the translator."""
        seg_key = hashlib.sha256(
            f"{source_lang}|{target_lang}|{text}".encode()
        ).hexdigest()
        if seg_key in self._translation_cache:
            return self._translation_cache[seg_key]

        src = self._translator_code(source_lang)
        tgt = self._translator_code(target_lang)

        # Capture loop variables explicitly to avoid closure pitfalls.
        backends = [
            (GoogleTranslator, src, tgt),
            (MyMemoryTranslator, src, tgt),
        ]

        for cls_, s, t in backends:
            try:
                translated = await asyncio.to_thread(
                    lambda c=cls_, s_=s, t_=t: c(source=s_, target=t_).translate(text)
                )
                if translated and translated.strip():
                    self._cache_put(seg_key, translated)
                    return translated
            except Exception as exc:
                print(f"[AutoTranslate] Backend {cls_.__name__} failed: {exc}")
                continue

        return text

    def _cache_put(self, key: str, value: str) -> None:
        if len(self._translation_cache) >= self._max_cache_size:
            keys = list(self._translation_cache.keys())
            for k in keys[: len(keys) // 2]:
                del self._translation_cache[k]
        self._translation_cache[key] = value

    # ------------------------------------------------------------------
    # Content helpers (plain string + multimodal list formats)
    # ------------------------------------------------------------------

    @staticmethod
    def _get_text_parts(content: Any) -> List[str]:
        if isinstance(content, str):
            return [content] if content else []
        if isinstance(content, list):
            return [
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            ]
        return []

    @staticmethod
    def _set_text_parts(message: dict, parts: List[str]) -> None:
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = parts[0] if parts else ""
        elif isinstance(content, list):
            idx = 0
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    if idx < len(parts):
                        item["text"] = parts[idx]
                        idx += 1

    # ------------------------------------------------------------------
    # inlet — runs BEFORE the LLM sees the request
    # ------------------------------------------------------------------

    async def inlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __metadata__: Optional[dict] = None,
        __event_emitter__: Optional[Callable] = None,
    ) -> dict:
        user_valves = (__user__ or {}).get("valves")
        if not self._is_user_enabled(user_valves):
            return body

        messages = body.get("messages", [])
        if not messages:
            return body

        # Detect language from the latest user message
        last_user_content = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                parts = self._get_text_parts(msg.get("content", ""))
                last_user_content = " ".join(parts).strip()
                if last_user_content:
                    break

        if not last_user_content:
            return body

        lang, confidence = self._detect_language(last_user_content)
        chat_id = self._resolve_chat_id(__metadata__, __user__, body)

        if lang == "en" or confidence < self.valves.detection_threshold:
            self._chat_languages.pop(chat_id, None)
            return body

        # Non-English detected
        self._store_chat_language(chat_id, lang)
        lang_name = LANGUAGE_MAP.get(lang, lang)

        await self._emit_status(
            __event_emitter__,
            f"Detected {lang_name} — translating to English...",
            done=False,
        )

        # Translate all user/assistant messages so the LLM gets fully-English
        # context.  Previous translations are cached — only the new message
        # costs an API call.
        for msg in messages:
            if msg.get("role") in ("user", "assistant"):
                parts = self._get_text_parts(msg.get("content", ""))
                translated_parts = []
                for part in parts:
                    if part.strip():
                        translated_parts.append(
                            await self._translate_text(part, lang, "en")
                        )
                    else:
                        translated_parts.append(part)
                if translated_parts:
                    self._set_text_parts(msg, translated_parts)

        await self._emit_status(
            __event_emitter__,
            f"Translated from {lang_name} to English",
            done=True,
        )

        return body

    # ------------------------------------------------------------------
    # outlet — runs AFTER the LLM response is assembled
    # ------------------------------------------------------------------

    async def outlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __metadata__: Optional[dict] = None,
        __event_emitter__: Optional[Callable] = None,
    ) -> dict:
        user_valves = (__user__ or {}).get("valves")
        if not self._is_user_enabled(user_valves):
            return body

        chat_id = self._resolve_chat_id(__metadata__, __user__, body)
        source_lang = self._chat_languages.get(chat_id)

        if not source_lang:
            return body

        messages = body.get("messages", [])

        last_assistant_msg = None
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                last_assistant_msg = msg
                break

        if not last_assistant_msg:
            return body

        parts = self._get_text_parts(last_assistant_msg.get("content", ""))
        if not any(p.strip() for p in parts):
            return body

        lang_name = LANGUAGE_MAP.get(source_lang, source_lang)

        await self._emit_status(
            __event_emitter__,
            f"Translating response to {lang_name}...",
            done=False,
        )

        translated_parts = []
        for part in parts:
            if part.strip():
                translated_parts.append(
                    await self._translate_text(part, "en", source_lang)
                )
            else:
                translated_parts.append(part)

        self._set_text_parts(last_assistant_msg, translated_parts)

        await self._emit_status(
            __event_emitter__,
            f"Translated to {lang_name}",
            done=True,
        )

        return body
