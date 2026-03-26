import io
import json
import os
import re
import uuid
import wave
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
from engine_utils.general_slicer import SliceContext, slice_data


class ASRConfig(HandlerBaseConfigModel, BaseModel):
    api_url: str = Field(default="https://api.siliconflow.cn/v1/audio/transcriptions")
    api_key: str = Field(default=os.getenv("SILICONFLOW_API_KEY"))
    model_name: str = Field(default="FunAudioLLM/SenseVoiceSmall")
    sample_rate: int = Field(default=16000)
    connect_timeout: float = Field(default=10.0)
    read_timeout: float = Field(default=120.0)


class ASRContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.output_audios = []
        self.audio_slice_context = SliceContext.create_numpy_slice_context(
            slice_size=16000,
            slice_axis=0,
        )
        self.shared_states = None
        self.dump_audio = True
        self.audio_dump_file = None
        if self.dump_audio:
            dump_file_path = os.path.join(DirectoryInfo.get_project_dir(), "dump_talk_audio.pcm")
            self.audio_dump_file = open(dump_file_path, "wb")


class HandlerASR(HandlerBase, ABC):
    def __init__(self):
        super().__init__()
        self.api_url = "https://api.siliconflow.cn/v1/audio/transcriptions"
        self.api_key = None
        self.model_name = "FunAudioLLM/SenseVoiceSmall"
        self.sample_rate = 16000
        self.connect_timeout = 10.0
        self.read_timeout = 120.0

    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(
            name="ASR_SiliconFlow",
            config_model=ASRConfig,
        )

    def get_handler_detail(self, session_context: SessionContext,
                           context: HandlerContext) -> HandlerDetail:
        definition = DataBundleDefinition()
        definition.add_entry(DataBundleEntry.create_audio_entry("avatar_audio", 1, 24000))
        inputs = {
            ChatDataType.HUMAN_AUDIO: HandlerDataInfo(
                type=ChatDataType.HUMAN_AUDIO,
            )
        }
        outputs = {
            ChatDataType.HUMAN_TEXT: HandlerDataInfo(
                type=ChatDataType.HUMAN_TEXT,
                definition=definition,
            )
        }
        return HandlerDetail(inputs=inputs, outputs=outputs)

    def load(self, engine_config: ChatEngineConfigModel, handler_config: Optional[BaseModel] = None):
        if not isinstance(handler_config, ASRConfig):
            handler_config = ASRConfig()
        self.api_url = handler_config.api_url
        self.api_key = handler_config.api_key
        self.model_name = handler_config.model_name
        self.sample_rate = handler_config.sample_rate
        self.connect_timeout = handler_config.connect_timeout
        self.read_timeout = handler_config.read_timeout

        if not self.api_key:
            raise RuntimeError("SiliconFlow ASR requires api_key or SILICONFLOW_API_KEY")

    def create_context(self, session_context, handler_config=None):
        context = ASRContext(session_context.session_info.session_id)
        context.shared_states = session_context.shared_states
        return context

    def start_context(self, session_context, handler_context):
        pass

    def _audio_to_wav_bytes(self, audio: np.ndarray) -> bytes:
        audio = np.asarray(audio).squeeze()
        audio = np.clip(audio, -1.0, 1.0)
        pcm = (audio * 32767.0).astype(np.int16)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(pcm.tobytes())
        return buffer.getvalue()

    def _build_multipart_body(self, wav_bytes: bytes) -> tuple[bytes, str]:
        boundary = f"----CodexBoundary{uuid.uuid4().hex}"
        body = bytearray()

        def add_form_field(name: str, value: str):
            body.extend(f"--{boundary}\r\n".encode("utf-8"))
            body.extend(
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8")
            )
            body.extend(value.encode("utf-8"))
            body.extend(b"\r\n")

        add_form_field("model", self.model_name)
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(b'Content-Disposition: form-data; name="file"; filename="speech.wav"\r\n')
        body.extend(b"Content-Type: audio/wav\r\n\r\n")
        body.extend(wav_bytes)
        body.extend(b"\r\n")
        body.extend(f"--{boundary}--\r\n".encode("utf-8"))
        return bytes(body), boundary

    def _transcribe(self, audio: np.ndarray) -> str:
        wav_bytes = self._audio_to_wav_bytes(audio)
        body, boundary = self._build_multipart_body(wav_bytes)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        }
        req = request.Request(self.api_url, data=body, headers=headers, method="POST")
        timeout = self.connect_timeout + self.read_timeout
        with request.urlopen(req, timeout=timeout) as response:
            resp = json.loads(response.read().decode("utf-8"))
        return resp.get("text", "")

    def handle(self, context: HandlerContext, inputs: ChatData,
               output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        output_definition = output_definitions.get(ChatDataType.HUMAN_TEXT).definition
        context = cast(ASRContext, context)
        if inputs.type == ChatDataType.HUMAN_AUDIO:
            audio = inputs.data.get_main_data()
        else:
            return
        speech_id = inputs.data.get_meta("speech_id")
        if speech_id is None:
            speech_id = context.session_id

        if audio is not None:
            audio = audio.squeeze()
            for audio_segment in slice_data(context.audio_slice_context, audio):
                if audio_segment is None or audio_segment.shape[0] == 0:
                    continue
                context.output_audios.append(audio_segment)

        speech_end = inputs.data.get_meta("human_speech_end", False)
        if not speech_end:
            return

        remainder_audio = context.audio_slice_context.flush()
        if remainder_audio is not None:
            if remainder_audio.shape[0] < context.audio_slice_context.slice_size:
                remainder_audio = np.concatenate(
                    [
                        remainder_audio,
                        np.zeros(
                            shape=(context.audio_slice_context.slice_size - remainder_audio.shape[0],),
                            dtype=remainder_audio.dtype,
                        ),
                    ]
                )
            context.output_audios.append(remainder_audio)

        if len(context.output_audios) == 0:
            return

        output_audio = np.concatenate(context.output_audios)
        context.output_audios.clear()

        if context.audio_dump_file is not None:
            context.audio_dump_file.write(output_audio.tobytes())

        try:
            output_text = self._transcribe(output_audio)
        except Exception as exc:
            logger.exception("SiliconFlow ASR request failed")
            output_text = ""

        output_text = re.sub(r"<\|.*?\|>", "", output_text).strip()
        if len(output_text) == 0:
            context.shared_states.enable_vad = True
            return

        output = DataBundle(output_definition)
        output.set_main_data(output_text)
        output.add_meta("human_text_end", False)
        output.add_meta("speech_id", speech_id)
        yield output

        end_output = DataBundle(output_definition)
        end_output.set_main_data("")
        end_output.add_meta("human_text_end", True)
        end_output.add_meta("speech_id", speech_id)
        yield end_output

    def destroy_context(self, context: HandlerContext):
        context = cast(ASRContext, context)
        if context.audio_dump_file is not None:
            context.audio_dump_file.close()
