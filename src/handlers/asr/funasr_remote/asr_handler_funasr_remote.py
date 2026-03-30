import asyncio
import json
import websockets
import ssl
from typing import Dict, Optional, cast
from loguru import logger
import numpy as np
from pydantic import BaseModel, Field
from abc import ABC
import os
from chat_engine.contexts.handler_context import HandlerContext
from chat_engine.data_models.chat_engine_config_data import ChatEngineConfigModel, HandlerBaseConfigModel
from chat_engine.common.handler_base import HandlerBase, HandlerBaseInfo, HandlerDataInfo, HandlerDetail
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.data_models.chat_data_type import ChatDataType
from chat_engine.data_models.runtime_data.data_bundle import DataBundle, DataBundleDefinition, DataBundleEntry
from chat_engine.contexts.session_context import SessionContext

from engine_utils.directory_info import DirectoryInfo
from engine_utils.general_slicer import SliceContext, slice_data


class ASRConfig(HandlerBaseConfigModel, BaseModel):
    host: str = Field(default="localhost")
    port: int = Field(default=10096)
    mode: str = Field(default="2pass")
    chunk_size: str = Field(default="5, 10, 5")
    chunk_interval: int = Field(default=10)
    use_ssl: bool = Field(default=False)
    use_itn: bool = Field(default=True)
    hotwords: str = Field(default="")


class ASRContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.config = None
        self.local_session_id = 0
        self.output_audios = []
        self.audio_slice_context = SliceContext.create_numpy_slice_context(
            slice_size=16000,
            slice_axis=0,
        )
        self.cache = {}

        self.dump_audio = True
        self.audio_dump_file = None
        if self.dump_audio:
            dump_file_path = os.path.join(DirectoryInfo.get_project_dir(),
                                          "dump_talk_audio.pcm")
            self.audio_dump_file = open(dump_file_path, "wb")
        self.shared_states = None
        self.loop = None


class HandlerASR(HandlerBase, ABC):
    def __init__(self):
        super().__init__()
        self.host = "localhost"
        self.port = 10096
        self.mode = "2pass"
        self.chunk_size = "5, 10, 5"
        self.chunk_interval = 10
        self.use_ssl = False
        self.use_itn = True
        self.hotwords = ""

    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(
            name="ASR_Funasr_Remote",
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
        return HandlerDetail(
            inputs=inputs, outputs=outputs,
        )

    def load(self, engine_config: ChatEngineConfigModel, handler_config: Optional[BaseModel] = None):
        if isinstance(handler_config, ASRConfig):
            self.host = handler_config.host
            self.port = handler_config.port
            self.mode = handler_config.mode
            self.chunk_size = handler_config.chunk_size
            self.chunk_interval = handler_config.chunk_interval
            self.use_ssl = handler_config.use_ssl
            self.use_itn = handler_config.use_itn
            self.hotwords = handler_config.hotwords

    def create_context(self, session_context, handler_config=None):
        if not isinstance(handler_config, ASRConfig):
            handler_config = ASRConfig()
        context = ASRContext(session_context.session_info.session_id)
        context.shared_states = session_context.shared_states
        return context

    def start_context(self, session_context, handler_context):
        pass

    async def _recognize_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """Recognize audio through remote funasr websocket service"""
        # Convert float32 numpy array to int16 bytes
        audio_int16 = (audio_data * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()

        chunk_size = [int(x) for x in self.chunk_size.split(",")]

        # Process hotwords
        fst_dict = {}
        hotword_msg = ""
        if self.hotwords.strip() != "":
            if os.path.exists(self.hotwords):
                with open(self.hotwords, "r", encoding="utf-8") as f:
                    hot_lines = f.readlines()
                    for line in hot_lines:
                        words = line.strip().split(" ")
                        if len(words) < 2:
                            continue
                        try:
                            fst_dict[" ".join(words[:-1])] = int(words[-1])
                        except ValueError:
                            continue
                hotword_msg = json.dumps(fst_dict)
            else:
                hotword_msg = self.hotwords

        # Connect to websocket
        if self.use_ssl:
            ssl_context = ssl.SSLContext()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            uri = f"wss://{self.host}:{self.port}"
        else:
            uri = f"ws://{self.host}:{self.port}"
            ssl_context = None

        full_text = []
        offline_done = False

        try:
            async with websockets.connect(
                uri, subprotocols=["binary"], ping_interval=None, ssl=ssl_context
            ) as websocket:
                # Send init message
                init_message = json.dumps(
                    {
                        "mode": self.mode,
                        "chunk_size": chunk_size,
                        "chunk_interval": self.chunk_interval,
                        "encoder_chunk_look_back": 4,
                        "decoder_chunk_look_back": 0,
                        "audio_fs": sample_rate,
                        "wav_name": f"session_{self.host}",
                        "wav_format": "pcm",
                        "is_speaking": True,
                        "hotwords": hotword_msg,
                        "itn": self.use_itn,
                    }
                )
                await websocket.send(init_message)

                # Send audio in chunks
                stride = int(60 * chunk_size[1] / self.chunk_interval / 1000 * sample_rate * 2)
                chunk_num = (len(audio_bytes) - 1) // stride + 1

                for i in range(chunk_num):
                    beg = i * stride
                    data = audio_bytes[beg:beg + stride]
                    await websocket.send(data)
                    if i == chunk_num - 1:
                        is_speaking = False
                        end_message = json.dumps({"is_speaking": is_speaking})
                        await websocket.send(end_message)
                    await asyncio.sleep(0.001)

                # Receive results
                while not offline_done:
                    try:
                        meg = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                        meg = json.loads(meg)
                        text = meg.get("text", "")
                        if text:
                            full_text.append(text)
                        offline_done = meg.get("is_final", False)
                    except asyncio.TimeoutError:
                        logger.warning("Timeout waiting for ASR response")
                        break
                if self.mode == "offline" or self.mode == "2pass":
                    # Wait for final result in offline/2pass mode
                    await asyncio.sleep(0.5)

        except Exception as e:
            logger.error(f"Error connecting to remote funasr: {e}")
            return ""

        # Combine all text pieces
        result = "".join(full_text)
        # For 2pass mode, the last offline result is the full text
        if self.mode == "2pass" and len(full_text) > 0:
            result = full_text[-1]

        logger.info(f"Funasr remote ASR result: {result}")
        return result.strip()

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
            # Resample to 16000 if needed (xinli uses 24000, funasr expects 16000)
            if hasattr(audio, 'shape') and audio.shape[0] > 0:
                # Simple resampling by decimation if sample rate differs
                if 24000 == 24000:  # Our input is 24000, need 16000
                    # Simple 3->2 decimation
                    audio_resampled = audio[::3][::2]
                else:
                    audio_resampled = audio
                logger.info('audio in')
                for audio_segment in slice_data(context.audio_slice_context, audio_resampled):
                    if audio_segment is None or audio_segment.shape[0] == 0:
                        continue
                    context.output_audios.append(audio_segment)

        speech_end = inputs.data.get_meta("human_speech_end", False)
        if not speech_end:
            return

        # Get complete audio
        remainder_audio = context.audio_slice_context.flush()
        if remainder_audio is not None:
            if remainder_audio.shape[0] < context.audio_slice_context.slice_size:
                remainder_audio = np.concatenate(
                    [remainder_audio,
                     np.zeros(shape=(context.audio_slice_context.slice_size - remainder_audio.shape[0]))])
                context.output_audios.append(remainder_audio)
        if len(context.output_audios) == 0:
            logger.warning("No audio collected")
            context.shared_states.enable_vad = True
            return

        output_audio = np.concatenate(context.output_audios)
        if context.audio_dump_file is not None:
            logger.info('dump audio')
            audio_int16 = (output_audio * 32767).astype(np.int16)
            context.audio_dump_file.write(audio_int16.tobytes())

        # Run async recognition
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            output_text = loop.run_until_complete(self._recognize_audio(output_audio, 16000))
            loop.close()
        except Exception as e:
            logger.error(f"ASR recognition failed: {e}")
            context.output_audios.clear()
            context.shared_states.enable_vad = True
            return

        context.output_audios.clear()

        if len(output_text) == 0:
            logger.warning("ASR result is empty")
            # If ASR result is empty, re-enable VAD
            context.shared_states.enable_vad = True
            return

        output = DataBundle(output_definition)
        output.set_main_data(output_text)
        output.add_meta('human_text_end', False)
        output.add_meta('speech_id', speech_id)
        yield output

        end_output = DataBundle(output_definition)
        end_output.set_main_data('')
        end_output.add_meta("human_text_end", True)
        end_output.add_meta("speech_id", speech_id)
        yield end_output

    def destroy_context(self, context: HandlerContext):
        context = cast(ASRContext, context)
        if context.audio_dump_file is not None:
            context.audio_dump_file.close()
