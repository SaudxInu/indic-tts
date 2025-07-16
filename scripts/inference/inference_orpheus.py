# * Install unsloth, SNAC, soundfile and loguru.
# pip install unsloth snac soundfile loguru

# Import necessary libraries.
# * unsloth import should always be at the top.
from unsloth import FastLanguageModel

from loguru import logger
from snac import SNAC
import soundfile as sf
import torch


# Set up constants and configurations.
HUGGINGFACE_USERNAME = ""  # ! Fill.
BASE_MODEL = f"{HUGGINGFACE_USERNAME}indic-tts-lora-tamil-stage-2"
MAX_SEQ_LENGTH = 2048
HUGGINGFACE_TOKEN = ""  # ! Fill.

# * Use the following command to run the inference: python inference_orpheus.py

# Load the model and tokenizer.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    load_in_4bit=True,
    max_seq_length=MAX_SEQ_LENGTH,
    token=HUGGINGFACE_TOKEN,
)
logger.success(f"Loaded model: {BASE_MODEL}")

# Load the special tokens for the tokenizer.
tokeniser_length = 128256

end_of_speech_id = tokeniser_length + 2
pad_token_id = tokeniser_length + 7
audio_start_id = tokeniser_length + 10

pad_token = tokenizer.decode([pad_token_id])

logger.success("Load special tokens for the tokenizer.")

# Wrap the model for inference.
FastLanguageModel.for_inference(model)
logger.success(f"{BASE_MODEL} is ready for inference.")

# Set the padding token and padding side.
tokenizer.pad_token = pad_token
tokenizer.padding_side = "left"
logger.success("Set padding token and padding side for the tokenizer.")

# Load the SNAC model for audio decoding.
snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
logger.success("Loaded SNAC model for audio decoding.")


# Function to generate audio.
def generate_audio(
    row,
    model,
    tokenizer,
    user=False,
    temperature=0.4,
    top_p=0.9,
    repetition_penalty=1.05,
):
    try:
        if user:
            prompt = row["eval_text_user"]
        else:
            prompt = row["eval_text_no_user"]
        inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        max_tokens = MAX_SEQ_LENGTH - inputs.input_ids.shape[1]
        output = model.generate(
            input_ids=inputs.input_ids.to("cuda"),
            attention_mask=inputs.attention_mask.to("cuda"),
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=end_of_speech_id,
        )
        audio_ids = []
        for id in output[0]:
            if id >= audio_start_id:
                audio_ids.append(id.item())
        clean_audio_ids = []
        for i in range((len(audio_ids) + 1) // 7):
            for j in range(7):
                clean_audio_ids += [audio_ids[7 * i + j] - audio_start_id]
        codes = [[], [], []]
        for i in range((len(clean_audio_ids) + 1) // 7):
            codes[0].append(clean_audio_ids[7 * i])
            codes[1].append(clean_audio_ids[7 * i + 1] - 4096)
            codes[2].append(clean_audio_ids[7 * i + 2] - (2 * 4096))
            codes[2].append(clean_audio_ids[7 * i + 3] - (3 * 4096))
            codes[1].append(clean_audio_ids[7 * i + 4] - (4 * 4096))
            codes[2].append(clean_audio_ids[7 * i + 5] - (5 * 4096))
            codes[2].append(clean_audio_ids[7 * i + 6] - (6 * 4096))
        codes = [
            torch.tensor(codes[0]).unsqueeze(0),
            torch.tensor(codes[1]).unsqueeze(0),
            torch.tensor(codes[2]).unsqueeze(0),
        ]
        audio = snac_model.decode(codes)
        return audio.detach().squeeze().to("cpu").numpy()
    except Exception as e:
        logger.error(f"Error decoding audio: {e}")
        return None


# Run inference.
# * Please refer to the training script to create prompt from SNAC tokens.
row = {
    "eval_text_user": "<custom_token_3><|begin_of_text|>tamil128: நான் பதினான்காம் தேதி காலை ரோம்ல இருக்கணும், ஆனா அதுக்கு முன்னாடி, பதிமூன்றாம் தேதி மாலையே போயிடனும் எனக்கு பிடிக்கும்னு.<|eot_id|><custom_token_4><custom_token_5><custom_token_1>"
}
eval_sample = generate_audio(row, model, tokenizer, True)
if eval_sample is None:
    logger.error("Failed to generate audio for evaluation sample.")
else:
    filename = "eval.wav"
    sf.write(filename, eval_sample, 24000)
    logger.success(f"Generated and saved evaluation sample audio as {filename}.")
