import base64
import json
import os
import re
import time
import uuid
from abc import ABC
from typing import Dict, Optional, cast
from urllib import request

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field

from chat_engine.common.handler_base import HandlerBase, HandlerBaseInfo, HandlerDataInfo, HandlerDetail
from chat_engine.contexts.handler_context import HandlerContext
from chat_engine.contexts.session_context import SessionContext
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.data_models.chat_data_type import ChatDataType
from chat_engine.data_models.chat_engine_config_data import ChatEngineConfigModel, HandlerBaseConfigModel
from chat_engine.data_models.runtime_data.data_bundle import DataBundle, DataBundleDefinition, DataBundleEntry
from engine_utils.directory_info import DirectoryInfo


class TTSConfig(HandlerBaseConfigModel, BaseModel):
    voice: str = Field(default="zh_female_shuangkuaisisi_moon_bigtts")
    sample_rate: int = Field(default=24000)
    audio_format: str = Field(default="pcm")
    app_id: str = Field(default=os.getenv("VOLCENGINE_TTS_APP_ID"))
    access_token: str = Field(default=os.getenv("VOLCENGINE_TTS_ACCESS_TOKEN"))
    resource_id: str = Field(default="seed-tts-2.0")
    api_url: str = Field(default="https://openspeech.bytedance.com/api/v3/tts/unidirectional/sse")
    connect_timeout: float = Field(default=10.0)
    read_timeout: float = Field(default=120.0)
    chunk_flush_bytes: int = Field(default=8000)


class TTSContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.input_text = ""
        self.dump_audio = False
        self.audio_dump_file = None


class HandlerTTS(HandlerBase, ABC):
    def __init__(self):
        super().__init__()
        self.voice = "zh_female_shuangkuaisisi_moon_bigtts"
        self.sample_rate = 24000
        self.audio_format = "pcm"
        self.app_id = None
        self.access_token = None
        self.resource_id = "seed-tts-2.0"
        self.api_url = "https://openspeech.bytedance.com/api/v3/tts/unidirectional/sse"
        self.connect_timeout = 10.0
        self.read_timeout = 120.0
        self.chunk_flush_bytes = 8000
        self.sentence_split_pattern = r"(?<=[.!?\u3002\uff01\uff1f])"
        self.allowed_text_pattern = r"[^a-zA-Z0-9\u4e00-\u9fff,.\~!?\u3002\uff01\uff1f\uff0c\uff1b\uff1a\u3001\s]"

    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(config_model=TTSConfig)

    def get_handler_detail(self, session_context: SessionContext,
                           context: HandlerContext) -> HandlerDetail:
        definition = DataBundleDefinition()
        definition.add_entry(DataBundleEntry.create_audio_entry("avatar_audio", 1, self.sample_rate))
        inputs = {
            ChatDataType.AVATAR_TEXT: HandlerDataInfo(
                type=ChatDataType.AVATAR_TEXT,
            )
        }
        outputs = {
            ChatDataType.AVATAR_AUDIO: HandlerDataInfo(
                type=ChatDataType.AVATAR_AUDIO,
                definition=definition,
            )
        }
        return HandlerDetail(inputs=inputs, outputs=outputs)

    def load(self, engine_config: ChatEngineConfigModel, handler_config: Optional[BaseModel] = None):
        if not isinstance(handler_config, TTSConfig):
            handler_config = TTSConfig()
        self.voice = handler_config.voice
        self.sample_rate = handler_config.sample_rate
        self.audio_format = handler_config.audio_format
        self.app_id = str(handler_config.app_id) if handler_config.app_id is not None else None
        self.access_token = handler_config.access_token
        self.resource_id = handler_config.resource_id
        self.api_url = handler_config.api_url
        self.connect_timeout = handler_config.connect_timeout
        self.read_timeout = handler_config.read_timeout
        self.chunk_flush_bytes = handler_config.chunk_flush_bytes

        if not self.app_id or not self.access_token:
            raise RuntimeError(
                "Volcengine TTS requires app_id/access_token or VOLCENGINE_TTS_APP_ID/VOLCENGINE_TTS_ACCESS_TOKEN"
            )

    def create_context(self, session_context, handler_config=None):
        context = TTSContext(session_context.session_info.session_id)
        if context.dump_audio:
            dump_file_path = os.path.join(
                DirectoryInfo.get_project_dir(),
                "temp",
                f"dump_avatar_audio_{context.session_id}_{time.localtime().tm_hour}_{time.localtime().tm_min}.pcm",
            )
            context.audio_dump_file = open(dump_file_path, "wb")
        return context

    def start_context(self, session_context, context: HandlerContext):
        pass

    def filter_text(self, text: str) -> str:
        return re.sub(self.allowed_text_pattern, "", text)

    def _submit_audio_chunk(self, context: TTSContext, output_definition, speech_id, pcm_bytes: bytes):
        if not pcm_bytes:
            return
        if context.audio_dump_file is not None:
            context.audio_dump_file.write(pcm_bytes)
        output_audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32767.0
        output_audio = output_audio[np.newaxis, ...]
        output = DataBundle(output_definition)
        output.set_main_data(output_audio)
        output.add_meta("avatar_speech_end", False)
        output.add_meta("speech_id", speech_id)
        context.submit_data(output)

    def _stream_tts(self, context: TTSContext, output_definition, speech_id, text: str):
        request_started_at = time.perf_counter()
        payload = {
            "user": {"uid": str(speech_id)},
            "req_params": {
                "text": text,
                "speaker": self.voice,
                "audio_params": {
                    "format": self.audio_format,
                    "sample_rate": self.sample_rate,
                },
            },
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "X-Api-App-Id": self.app_id,
            "X-Api-Access-Key": self.access_token,
            "X-Api-Resource-Id": self.resource_id,
            "X-Api-Request-Id": str(uuid.uuid4()),
        }
        req = request.Request(self.api_url, data=body, headers=headers, method="POST")
        timeout = self.connect_timeout + self.read_timeout
        audio_buffer = bytearray()
        first_audio_packet_at = None
        first_submit_at = None
        total_audio_bytes = 0

        with request.urlopen(req, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line or not line.startswith("data:"):
                    continue
                payload = json.loads(line[5:].strip())
                code = payload.get("code")
                if code not in (0, 20000000):
                    raise RuntimeError(f"Volcengine TTS failed: {payload}")
                data = payload.get("data")
                if data:
                    decoded_audio = base64.b64decode(data)
                    if first_audio_packet_at is None:
                        first_audio_packet_at = time.perf_counter()
                        logger.info(
                            "volcengine tts first audio packet in {:.3f}s for text: {}",
                            first_audio_packet_at - request_started_at,
                            text,
                        )
                    total_audio_bytes += len(decoded_audio)
                    audio_buffer.extend(decoded_audio)
                    if len(audio_buffer) >= self.chunk_flush_bytes:
                        if first_submit_at is None:
                            first_submit_at = time.perf_counter()
                            logger.info(
                                "volcengine tts first submit in {:.3f}s, buffered {} bytes for text: {}",
                                first_submit_at - request_started_at,
                                len(audio_buffer),
                                text,
                            )
                        self._submit_audio_chunk(context, output_definition, speech_id, bytes(audio_buffer))
                        audio_buffer.clear()

        if len(audio_buffer) > 0:
            if first_submit_at is None:
                first_submit_at = time.perf_counter()
                logger.info(
                    "volcengine tts first submit in {:.3f}s, buffered {} bytes for text: {}",
                    first_submit_at - request_started_at,
                    len(audio_buffer),
                    text,
                )
            self._submit_audio_chunk(context, output_definition, speech_id, bytes(audio_buffer))

        logger.info(
            "volcengine tts completed in {:.3f}s, total_audio_bytes={}, text={}",
            time.perf_counter() - request_started_at,
            total_audio_bytes,
            text,
        )

    def _emit_speech_end(self, context: TTSContext, output_definition, speech_id):
        output = DataBundle(output_definition)
        output.set_main_data(np.zeros(shape=(1, 240), dtype=np.float32))
        output.add_meta("avatar_speech_end", True)
        output.add_meta("speech_id", speech_id)
        context.submit_data(output)

    def handle(self, context: HandlerContext, inputs: ChatData,
               output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        output_definition = output_definitions.get(ChatDataType.AVATAR_AUDIO).definition
        context = cast(TTSContext, context)
        if inputs.type == ChatDataType.AVATAR_TEXT:
            text = inputs.data.get_main_data()
        else:
            return
        speech_id = inputs.data.get_meta("speech_id")
        if speech_id is None:
            speech_id = context.session_id

        if text is not None:
            text = re.sub(r"<\|.*?\|>", "", text)
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
            text = text.replace("<think>", "").replace("</think>", "")
            context.input_text += self.filter_text(text)

        text_end = inputs.data.get_meta("avatar_text_end", False)
        try:
            if not text_end:
                sentences = re.split(self.sentence_split_pattern, context.input_text)
                if len(sentences) > 1:
                    complete_sentences = sentences[:-1]
                    context.input_text = sentences[-1]
                    for sentence in complete_sentences:
                        sentence = sentence.strip()
                        if len(sentence) < 2:
                            continue
                        logger.info("volcengine tts sentence chunk: {}", sentence)
                        self._stream_tts(context, output_definition, speech_id, sentence)
            else:
                final_text = context.input_text.strip()
                if final_text:
                    logger.info("volcengine tts final chunk: {}", final_text)
                    self._stream_tts(context, output_definition, speech_id, final_text)
                context.input_text = ""
                self._emit_speech_end(context, output_definition, speech_id)
        except Exception:
            logger.exception("Volcengine TTS request failed")
            context.input_text = ""
            self._emit_speech_end(context, output_definition, speech_id)

    def destroy_context(self, context: HandlerContext):
        context = cast(TTSContext, context)
        if context.audio_dump_file is not None:
            context.audio_dump_file.close()
