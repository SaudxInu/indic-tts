# * Install Modal.
# uv run pip install modal

# * Setup Modal.
# uv run modal setup

# * Run to deploy the Modal app.
# uv run modal deploy scripts/modal/snorTTS_Indic_v0_server.py

# * Test.
# curl -X 'POST' \
#   'https://snorbyte--snortts-indic-v0-server-prod-ttsserver-serve.modal.run/?utterance=%E0%A4%95%E0%A4%B2%20%E0%A4%AE%E0%A5%88%E0%A4%82%E0%A4%A8%E0%A5%87%20%E0%A4%B8%E0%A4%BF%E0%A4%B0%E0%A5%8D%E0%A4%AB%20%E2%82%B9500%20%E0%A4%AE%E0%A5%87%E0%A4%82%20%E0%A4%8F%E0%A4%95%20cool%20headphones%20%E0%A4%B2%E0%A5%87%20%E0%A4%B2%E0%A4%BF%E0%A4%8F%2C%20%E0%A4%AC%E0%A4%B9%E0%A5%81%E0%A4%A4%20%E0%A4%AC%E0%A4%A2%E0%A4%BC%E0%A4%BF%E0%A4%AF%E0%A4%BE%20deal%20%E0%A4%A5%E0%A4%BE%20%E0%A4%AF%E0%A4%BE%E0%A4%B0%21&user_id=159&language=hindi&temperature=0.4&top_p=0.9&repetition_penalty=1.05&speed=1.05&denoise=true&stream=false' \
#   -H 'accept: audio/mpeg' \
#   -d '' \
#   --output outputs/output_non_stream.mp3

# curl -X 'POST' \
#   'https://snorbyte--snortts-indic-v0-server-prod-ttsserver-serve.modal.run/?utterance=%E0%A4%95%E0%A4%B2%20%E0%A4%AE%E0%A5%88%E0%A4%82%E0%A4%A8%E0%A5%87%20%E0%A4%B8%E0%A4%BF%E0%A4%B0%E0%A5%8D%E0%A4%AB%20%E2%82%B9500%20%E0%A4%AE%E0%A5%87%E0%A4%82%20%E0%A4%8F%E0%A4%95%20cool%20headphones%20%E0%A4%B2%E0%A5%87%20%E0%A4%B2%E0%A4%BF%E0%A4%8F%2C%20%E0%A4%AC%E0%A4%B9%E0%A5%81%E0%A4%A4%20%E0%A4%AC%E0%A4%A2%E0%A4%BC%E0%A4%BF%E0%A4%AF%E0%A4%BE%20deal%20%E0%A4%A5%E0%A4%BE%20%E0%A4%AF%E0%A4%BE%E0%A4%B0%21&user_id=159&language=hindi&temperature=0.4&top_p=0.9&repetition_penalty=1.05&speed=1.05&denoise=true&stream=true' \
#   -H 'accept: audio/mpeg' \
#   -d '' \
#   --output outputs/output_stream.mp3

# Import Modal.
import modal


# Define constants.
APP_NAME = "snorTTS-Indic-v0-server-dev"
SCALEDOWN_WINDOW = 15 * 60
TIMEOUT = 10 * 60
MIN_CONTAINERS = 1
MAX_CONTAINERS = 1
MAX_CONCURRENT_REQUESTS = 5


# Define the Docker image.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "curl",  # Install curl for downloading files.
        "ffmpeg",  # Install ffmpeg for audio processing.
        "git",  # Install git for version control.
        "libsox-dev",  # Install SoX for audio processing.
    )
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",  # Install Rust.
    )
    .env(
        {
            "PATH": "/root/.cargo/bin:${PATH}",  # Add Rust to PATH.
            "HF_HUB_ENABLE_HF_TRANSFER": "1",  # Set `HF_HUB_ENABLE_HF_TRANSFER` for fast model transfers.
        }
    )
    .pip_install(
        "deepfilternet",  # Install DeepFilterNet for audio denoising.
        "fastapi[standard]",  # Install FastAPI for building the API.
        "hf_transfer",  # Install Hugging Face transfer for fast model transfers.
        "loguru",  # Install Loguru for logging.
        "numpy",  # Install NumPy for numerical operations.
        "pydub",  # Install Pydub for audio processing.
        "snac",  # Install SNAC for audio decoding.
        "torchaudio",  # Install Torchaudio for audio processing.
        "transformers",  # Install Transformers for model handling.
    )
)

# Create the Modal app.
app = modal.App(APP_NAME, image=image)

with image.imports():
    # Import necessary libraries for the remote function.
    from typing import Any
    import aiohttp
    import io
    import json

    from df.enhance import init_df, enhance
    from fastapi.responses import StreamingResponse
    from loguru import logger
    from pydub import AudioSegment
    from snac import SNAC
    from transformers import AutoTokenizer
    import numpy as np
    import torch
    import torchaudio as ta


@app.cls(
    cpu=4.0,  # Set number of CPU cores.
    memory=8192,  # Set memory in MiB.
    scaledown_window=SCALEDOWN_WINDOW,  # Set how long should we stay up with no requests.
    timeout=TIMEOUT,  # Set the timeout for the function.
    enable_memory_snapshot=True,  # Enable memory snapshot for better cold boot times.
    min_containers=MIN_CONTAINERS,  # Minimum number of containers to keep running.
    max_containers=MAX_CONTAINERS,  # Maximum number of containers to run.
    region="ap-south-1",  # Set the region for the function.
)
@modal.concurrent(
    max_inputs=MAX_CONCURRENT_REQUESTS
)  # Limit the number of concurrent requests.
class TTSServer:
    @modal.enter()
    def load(self) -> None:
        # Load the tokenizer.
        self.tokenizer = AutoTokenizer.from_pretrained("snorbyte/snorTTS-Indic-v0")
        logger.success("Loaded tokenizer from snorbyte/snorTTS-Indic-v0.")

        # Token related bookkeeping.
        # Set the tokenizer length.
        self.tokeniser_length = 128256
        logger.success("Set tokenizer length.")

        # Set the end of speech ID, pad token ID, and audio start ID.
        self.end_of_speech_id = self.tokeniser_length + 2
        self.pad_token_id = self.tokeniser_length + 7
        self.audio_start_id = self.tokeniser_length + 10
        logger.success("Set end of speech ID, pad token ID, and audio start ID.")

        # Decode the pad token.
        self.pad_token = self.tokenizer.decode([self.pad_token_id])
        logger.success("Decoded pad token.")

        # Set the padding token and padding side.
        self.tokenizer.pad_token = self.pad_token
        self.tokenizer.padding_side = "left"
        logger.success("Set padding token and padding side for the tokenizer.")

        # Models.
        # Load the SNAC model for audio decoding.
        self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
        logger.success("Loaded SNAC model for audio decoding.")

        # Initialize the DF model for audio denoising.
        self.df_model, self.df_state, _ = init_df()
        logger.success("Initialized DF model for audio denoising.")

    async def _decode_audio(self, audio_ids: list[int], speed: float, denoise: bool):
        # Offset the audio tokens by the audio start ID.
        snac_audio_ids = []
        for i in range((len(audio_ids) + 1) // 7):
            for j in range(7):
                snac_audio_ids += [audio_ids[7 * i + j] - self.audio_start_id]

        # Prepare the codes for SNAC decoding.
        # ! Please note: codes cannot be negative. If the model generates incorrect codes
        # ! at the wrong positions, audio generation will fail.
        codes = [[], [], []]
        for i in range((len(snac_audio_ids) + 1) // 7):
            codes[0].append(snac_audio_ids[7 * i])
            codes[1].append(snac_audio_ids[7 * i + 1] - 4096)
            codes[2].append(snac_audio_ids[7 * i + 2] - (2 * 4096))
            codes[2].append(snac_audio_ids[7 * i + 3] - (3 * 4096))
            codes[1].append(snac_audio_ids[7 * i + 4] - (4 * 4096))
            codes[2].append(snac_audio_ids[7 * i + 5] - (5 * 4096))
            codes[2].append(snac_audio_ids[7 * i + 6] - (6 * 4096))
        codes = [
            torch.tensor(codes[0]).unsqueeze(0),
            torch.tensor(codes[1]).unsqueeze(0),
            torch.tensor(codes[2]).unsqueeze(0),
        ]

        try:
            # Decode the audio using SNAC.
            audio = self.snac_model.decode(codes).reshape(1, -1)
            logger.success(f"Decoded {len(snac_audio_ids)} SNAC tokens to audio.")
        except Exception as e:
            logger.error(f"Error decoding audio: {e}")
            return None

        # Speed up or slow down the audio.
        if abs(speed - 1.0) > 1e-4:
            try:
                audio, _ = ta.sox_effects.apply_effects_tensor(
                    audio, 24_000, effects=[["tempo", f"{speed}"]]
                )
                logger.success(
                    f"Applied speed effect to audio with speed factor {speed}."
                )
            except Exception as e:
                logger.error(f"Error applying speed effect: {e}")
                return None

        # Denoise the audio.
        if denoise:
            try:
                audio = ta.transforms.Resample(orig_freq=24_000, new_freq=48_000)(audio)
                audio = enhance(self.df_model, self.df_state, audio)
                audio = ta.transforms.Resample(orig_freq=48_000, new_freq=24_000)(audio)
                logger.success("Denoised audio using DeepFilterNet.")
            except Exception as e:
                logger.error(f"Error denoising audio: {e}")
                return None

        # Move the audio to CPU and convert to numpy array.
        audio = audio.detach().squeeze().cpu().numpy()

        return audio

    async def _generate(
        self,
        utterance: str,
        user_id: str = 159,
        language: str = "hindi",
        temperature: float = 0.4,
        top_p: float = 0.9,
        repetition_penalty: float = 1.05,
        speed: float = 1.05,
        denoise: bool = False,
        stream: bool = True,
    ):
        try:
            # Limit the utterance length to 50 words.
            utterance = " ".join(utterance.split(" ")[:50])

            logger.info(
                f"Generating audio for utterance, {utterance}, user_id, {user_id}, language, {language}, "
                f"temperature, {temperature}, top_p, {top_p}, repetition_penalty, {repetition_penalty}, "
                f"speed, {speed}, denoise, {denoise} and stream, {stream}."
            )

            # Create the prompt.
            prompt = f"<custom_token_3><|begin_of_text|>{language}{user_id}: {utterance}<|eot_id|><custom_token_4><custom_token_5><custom_token_1>"

            # Tokenize the prompt.
            inputs = self.tokenizer(prompt, add_special_tokens=False)

            # Set max audio tokens to generate.
            max_tokens = 2048 - len(inputs.input_ids)

            # Generate the output.
            async with aiohttp.ClientSession(
                base_url="https://snorbyte--snortts-indic-v0-vllm-dev-serve.modal.run"
            ) as session:
                # Prepare the payload for the vLLM server.
                # ! Without type hinting the vLLM server will not recognize the request.
                payload: dict[str, Any] = {
                    "prompt": prompt,
                    "model": "llm",
                    "stream": True,
                    "temperature": 0.4,
                    "top_p": 0.9,
                    "max_tokens": max_tokens,
                    "repetition_penalty": 1.05,
                    "add_special_tokens": False,
                    "stop_token_ids": [128258],
                }

                # Set the headers for the request.
                headers = {
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                }

                # Initialize the audio tokens list.
                audio_ids = []

                # Send the request to the vLLM server to generate audio.
                async with session.post(
                    "/v1/completions",
                    json=payload,
                    headers=headers,
                    timeout=1 * 60,
                ) as resp:
                    # Maintine a buffer for the audio data.
                    buffer = io.BytesIO()

                    # Stream the vLLM response.
                    async for raw in resp.content:
                        # Check if the response is successful.
                        resp.raise_for_status()

                        # Decode bytes.
                        line = raw.decode().strip()

                        # Skip empty lines and end of stream.
                        if not line or line == "data: [DONE]":
                            continue

                        # Remove the "data: " prefix if present.
                        if line.startswith("data: "):
                            line = line[len("data: ") :]

                        # Parse the JSON response.
                        chunk = json.loads(line)

                        # Tokenize the generated tokens.
                        output = self.tokenizer(
                            chunk["choices"][0]["text"], add_special_tokens=False
                        ).input_ids

                        # Extract audio tokens from the output.
                        for id in output:
                            if id >= self.audio_start_id:
                                audio_ids.append(id)

                        # If streaming is enabled and the audio_ids list has more than 168 tokens,
                        # decode and yield the audio.
                        # ! This will lead to jittering in the audio stream.
                        if stream and len(audio_ids) > 168:
                            # Decode tokens to audio.
                            audio = await self._decode_audio(audio_ids, speed, denoise)

                            if audio is not None:
                                # Write the audio to the buffer.
                                # Convert to int16 PCM format expected by AudioSegment.
                                audio_int16 = (audio * 32767).astype(np.int16)

                                # Create raw audio segment.
                                raw_audio = AudioSegment(
                                    audio_int16.tobytes(),
                                    frame_rate=24000,
                                    sample_width=2,
                                    channels=1,
                                )

                                # Export the audio to the buffer in MP3 format.
                                raw_audio.export(buffer, format="mp3", bitrate="96k")

                                # Reset the buffer's internal pointer to the beginning of the stream.
                                # This allows reading the entire content from the start.
                                buffer.seek(0)

                                # Read the entire contents of the buffer into the `data` variable.
                                audio_data = buffer.read()

                                # Move the buffer's internal pointer back to the beginning again.
                                # This is done to prepare it for clearing.
                                buffer.seek(0)

                                # Truncate the buffer, effectively removing all contents.
                                # This clears it for reuse with new audio data.
                                buffer.truncate(0)

                                # Yield the audio data.
                                yield audio_data

                                # Keep the last incomplete frame.
                                last_index = len(audio_ids) % 7
                                if last_index == 0:
                                    audio_ids = []
                                else:
                                    audio_ids = audio_ids[-last_index:]

                    # Check if there are any remaining audio tokens to process.
                    if audio_ids:
                        # Decode tokens to audio.
                        audio = await self._decode_audio(audio_ids, speed, denoise)

                        if audio is not None:
                            # Write the audio to the buffer.
                            # Convert to int16 PCM format expected by AudioSegment.
                            audio_int16 = (audio * 32767).astype(np.int16)

                            # Create raw audio segment.
                            raw_audio = AudioSegment(
                                audio_int16.tobytes(),
                                frame_rate=24000,
                                sample_width=2,
                                channels=1,
                            )

                            # Export the audio to the buffer in MP3 format.
                            raw_audio.export(buffer, format="mp3", bitrate="96k")

                            # Reset the buffer's internal pointer to the beginning of the stream.
                            # This allows reading the entire content from the start.
                            buffer.seek(0)

                            # Read the entire contents of the buffer into the `data` variable.
                            audio_data = buffer.read()

                            # Move the buffer's internal pointer back to the beginning again.
                            # This is done to prepare it for clearing.
                            buffer.seek(0)

                            # Truncate the buffer, effectively removing all contents.
                            # This clears it for reuse with new audio data.
                            buffer.truncate(0)

                            # Yield the audio data.
                            yield audio_data
        except Exception as e:
            logger.exception(f"Error during audio generation: {e}")

    @modal.fastapi_endpoint(
        docs=True, method="POST"
    )  # Define a FastAPI endpoint for TTS.
    async def serve(
        self,
        utterance: str,
        user_id: str = 159,
        language: str = "hindi",
        temperature: float = 0.4,
        top_p: float = 0.9,
        repetition_penalty: float = 1.05,
        speed: float = 1.05,
        denoise: bool = False,
        stream: bool = True,
    ):
        # Stream the generated audio as an MP3 response.
        return StreamingResponse(
            self._generate(
                utterance,
                user_id=user_id,
                language=language,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                speed=speed,
                denoise=denoise,
                stream=stream,
            ),
            media_type="audio/mpeg",
        )
