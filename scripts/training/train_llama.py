# * This script was not rigorously tested, so it may not work as expected. We would suggest to
# * edit the script to follow Orpheus training script.

# * Install unsloth, PEFT, Weights & Biases, SNAC, pandas, soundfile and loguru.
# pip install unsloth peft==0.15.2 wandb snac pandas soundfile loguru

# * Login to Weights & Biases.
# wandb login

# Import necessary libraries.
# * unsloth import should always be at the top.
from unsloth import FastLanguageModel

import os

from datasets import load_dataset
from huggingface_hub import login
from loguru import logger
from snac import SNAC
from trl import SFTConfig, SFTTrainer
import soundfile as sf
import torch
import wandb


# Set up constants and configurations.
HUGGINGFACE_USERNAME = ""  # ! Fill.
BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
TRAIN_CSV_PATH = "data/data_stage_1.csv"
VALID_CSV_PATH = "data/data_eval_stage_2.csv"
TRAIN_NUM_SAMPLES = None
EVAL_NUM_SAMPLES = None
MAX_SEQ_LENGTH = 2048
N_CODEBOOKS, CODEBOOK_SIZE = 3, 4096
FIELDS = [
    "user",
    "gender",
    "age",
    "language",
    "utterance",
    "audio",
]
START_OF_SPECIAL_TOKENS = {field: f"<|start_of_{field}|>" for field in FIELDS}
END_OF_SPECIAL_TOKENS = {field: f"<|end_of_{field}|>" for field in FIELDS}
SNAC_TOKENS = [
    f"<|snac_{i}_{j}|>" for i in range(N_CODEBOOKS) for j in range(CODEBOOK_SIZE)
]
PER_DEVICE_TRAIN_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4
FULL_FINETUNING = True  # Set to False for LoRA training.
MODEL_NAME = "indic-tts-lora-training"
WANDB_USERNAME = ""  # ! Fill.
WANDB_PROJECT = "indic-tts-lora-training"
WANDB_LOG_MODEL = "checkpoint"
WANDB_RUN_NAME = None
WANDB_RUN_ID = None
SEED = 3407
HUGGINGFACE_TOKEN = ""  # ! Fill.
WANDB_TOKEN = ""  # ! Fill.

# * Use the following command to start the training: python train_llama.py

# Login to Hugging Face.
login(token=HUGGINGFACE_TOKEN)

# Login to Weights & Biases.
wandb.login(key=WANDB_TOKEN)

# Set up environment variables for Weights & Biases.
os.environ["WANDB_PROJECT"] = WANDB_PROJECT
os.environ["WANDB_LOG_MODEL"] = WANDB_LOG_MODEL

# Load the model and tokenizer.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    load_in_4bit=not FULL_FINETUNING,
    max_seq_length=MAX_SEQ_LENGTH,
    full_finetuning=FULL_FINETUNING,
)
logger.success(f"Loaded model: {BASE_MODEL}")

# Set the end of sequence token.
EOS_TOKEN = tokenizer.eos_token

# Add new special tokens to the tokenizer.
new_special_tokens = (
    list(START_OF_SPECIAL_TOKENS.values())
    + list(END_OF_SPECIAL_TOKENS.values())
    + SNAC_TOKENS
)
tokenizer.add_tokens(new_special_tokens, special_tokens=True)
model.resize_token_embeddings(len(tokenizer))
snac_offset = len(tokenizer.get_vocab()) - len(SNAC_TOKENS)
logger.success("Added new special tokens to the tokenizer.")

if not FULL_FINETUNING:
    # Get parameter efficient fine-tuning model.
    model = FastLanguageModel.get_peft_model(
        model,
        r=192,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
            "lm_head",
            "embed_tokens",
        ],
        lora_alpha=384,
        random_state=SEED,
    )
    logger.success("Initialized parameter efficient fine-tuning model.")

# Load training and validation datasets.
# The dataset should be in CSV format with columns user (str), language (str), utterance (str), and snac_codes (list).
train_dataset = load_dataset("csv", data_files=TRAIN_CSV_PATH)["train"]
eval_dataset = load_dataset("csv", data_files=VALID_CSV_PATH)["train"]

if TRAIN_NUM_SAMPLES:
    train_dataset = train_dataset.shuffle(seed=SEED).select(
        range(min(TRAIN_NUM_SAMPLES, len(train_dataset)))
    )

if EVAL_NUM_SAMPLES:
    eval_dataset = eval_dataset.shuffle(seed=SEED).select(
        range(min(EVAL_NUM_SAMPLES, len(eval_dataset)))
    )

logger.success(
    f"Loaded datasets: {len(train_dataset)} training samples, {len(eval_dataset)} evaluation samples."
)


# Format SNAC audio codes.
def format_snac_audio_codes(row):
    audio_codes = row["snac_codes"]
    if isinstance(audio_codes, str):
        audio_codes = eval(audio_codes)
    snac_tokens = [[], [], []]
    for i, layer in enumerate(audio_codes):
        for code in layer:
            snac_tokens[i].append(f"<|snac_{i}_{code}|>")
    row["snac_tokens"] = snac_tokens
    return row


train_dataset = train_dataset.map(format_snac_audio_codes)
eval_dataset = eval_dataset.map(format_snac_audio_codes)
logger.success("Formatted SNAC audio codes.")


# Flatten SNAC audio codes.
def flatten_audio_codes(row):
    audio_codes = row["snac_tokens"]
    flattened_codes = []
    for i in range(len(audio_codes[0])):
        flattened_codes.append(audio_codes[0][i])
        flattened_codes.append(audio_codes[1][2 * i])
        flattened_codes.append(audio_codes[2][4 * i])
        flattened_codes.append(audio_codes[2][(4 * i) + 1])
        flattened_codes.append(audio_codes[1][(2 * i) + 1])
        flattened_codes.append(audio_codes[2][(4 * i) + 2])
        flattened_codes.append(audio_codes[2][(4 * i) + 3])
    row["snac_tokens_list"] = flattened_codes
    return row


train_dataset = train_dataset.map(flatten_audio_codes)
eval_dataset = eval_dataset.map(flatten_audio_codes)
logger.success("Flattened SNAC audio codes.")


# Remove duplicate frames from the audio codes.
def remove_duplicate_frames(row):
    vals = row["snac_tokens_list"]
    if len(vals) % 7 != 0:
        raise ValueError("Input list length must be divisible by 7")
    result = vals[:7]
    for i in range(7, len(vals), 7):
        current_first = vals[i]
        previous_first = result[-7]
        if current_first != previous_first:
            result.extend(vals[i : i + 7])
    row["snac_tokens_list"] = result
    return row


train_dataset = train_dataset.map(remove_duplicate_frames)
eval_dataset = eval_dataset.map(remove_duplicate_frames)
logger.success("Removed duplicate frames from audio codes.")


# Define a function to format the prompt for each row in the dataset.
def format_text(row):
    input_parts = ""
    output_part = ""
    for field in FIELDS:
        if field != "audio":
            part = f"{START_OF_SPECIAL_TOKENS[field]} {row[field]} {END_OF_SPECIAL_TOKENS[field]}"
            input_parts += part + " "
        else:
            output_part = f"{START_OF_SPECIAL_TOKENS[field]} {' '.join(row['snac_tokens_list'])} {END_OF_SPECIAL_TOKENS[field]}"
    text = f"{input_parts.strip()} {output_part} {EOS_TOKEN}"
    eval_text = f"{input_parts.strip()} {START_OF_SPECIAL_TOKENS['audio']} "
    row["text"] = text
    row["eval_text"] = eval_text
    return row


train_dataset = train_dataset.map(format_text)
eval_dataset = eval_dataset.map(format_text)
logger.success("Formatted text for training and evaluation datasets.")

# Set training arguments.
training_args = SFTConfig(
    num_train_epochs=2,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    optim="adamw_8bit",
    learning_rate=5e-5 if FULL_FINETUNING else 2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.02,
    do_eval=True,
    eval_strategy="steps",
    eval_steps=50,
    logging_strategy="steps",
    logging_steps=1,
    save_strategy="steps",
    save_only_model=True,
    save_steps=1250,
    output_dir="outputs",
    report_to="wandb",
    run_name=WANDB_RUN_NAME,
    seed=SEED,
)

# Initialize the SFTTrainer.
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=2,
    packing=True,
    args=training_args,
)

logger.success("Initialized SFTTrainer with the specified configuration.")

# Start the training process.
logger.info("Starting the training process...")

run = wandb.init()

if WANDB_RUN_ID:
    logger.info(f"Resuming from Weights & Biases run ID: {WANDB_RUN_ID}")

    artifact = run.use_artifact(
        f"{WANDB_USERNAME}/{WANDB_PROJECT}/{WANDB_RUN_ID}", type="model"
    )

    artifact_dir = artifact.download()

    trainer.train(resume_from_checkpoint=artifact_dir)
else:
    try:
        logger.info("Attempting to resume training from the last checkpoint...")

        trainer.train(resume_from_checkpoint=True)
    except Exception as err:
        trainer.train()

# Finish the Weights & Biases run.
wandb.finish()

logger.success("Training completed successfully.")

# ! Saving and loading model doesn't work.
# # Save the model and tokenizer.
# model.save_pretrained_merged(
#     f"{HUGGINGFACE_USERNAME}/{MODEL_NAME}",
#     tokenizer,
#     save_method="merged_16bit",
# )
# logger.success("Saved the model and tokenizer locally.")

# model.push_to_hub_merged(
#     f"{HUGGINGFACE_USERNAME}/{MODEL_NAME}",
#     tokenizer,
#     save_method="merged_16bit",
#     token=HUGGINGFACE_TOKEN,
# )
# logger.success("Pushed the model and tokenizer to the Hugging Face Hub.")

# del trainer, model, tokenizer

# # Inference with the trained model.
# # Load the model and tokenizer.
# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name=f"{HUGGINGFACE_USERNAME}/{MODEL_NAME}",
#     load_in_4bit=True,
#     max_seq_length=MAX_SEQ_LENGTH,
# )

FastLanguageModel.for_inference(model)

logger.success(f"Loaded model for inference: {HUGGINGFACE_USERNAME}/{MODEL_NAME}")

# Load the SNAC model for audio decoding.
snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
logger.success("Loaded SNAC model for audio decoding.")


# Function to generate audio from a dataset row.
def generate_audio(
    row, model, tokenizer, temperature=0.4, top_p=0.9, repetition_penalty=1.05
):
    prompt = row["eval_text"]
    inputs = tokenizer(prompt, return_tensors="pt")
    max_tokens = MAX_SEQ_LENGTH - inputs.input_ids.shape[1]
    output = model.generate(
        input_ids=inputs.input_ids.to("cuda"),
        attention_mask=inputs.attention_mask.to("cuda"),
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    audio_ids = []
    for id in output[0]:
        if id >= snac_offset:
            audio_ids.append(id.item())
    clean_audio_ids = []
    for i in range((len(audio_ids) + 1) // 7):
        for j in range(7):
            clean_audio_ids += [audio_ids[7 * i + j], 220]
    audio_tokens = tokenizer.decode(clean_audio_ids).strip().split(" ")
    codes = [[], [], []]
    for i in range((len(audio_tokens) + 1) // 7):
        frame = []
        for j in range(7):
            _, _, code = audio_tokens[7 * i + j].split("_")
            code = int(code[:-2])
            frame.append(code)
        codes[0].append(frame[0])
        codes[1].append(frame[1])
        codes[2].append(frame[2])
        codes[2].append(frame[3])
        codes[1].append(frame[4])
        codes[2].append(frame[5])
        codes[2].append(frame[6])
    codes = [
        torch.tensor(codes[0]).unsqueeze(0),
        torch.tensor(codes[1]).unsqueeze(0),
        torch.tensor(codes[2]).unsqueeze(0),
    ]
    try:
        audio = snac_model.decode(codes)
    except Exception as e:
        logger.error(f"Error decoding audio: {e}")
        return None
    return audio.detach().squeeze().to("cpu").numpy()


# Generate and save some examples.
train_sample = generate_audio(train_dataset[0], model, tokenizer)
if train_sample is None:
    logger.error("Failed to generate audio for training sample.")
else:
    sf.write("train.wav", train_sample, 24000)
    logger.success("Generated and saved training sample audio.")

eval_sample = generate_audio(eval_dataset[0], model, tokenizer)
if eval_sample is None:
    logger.error("Failed to generate audio for evaluation sample.")
else:
    sf.write("eval.wav", eval_sample, 24000)
    logger.success("Generated and saved evaluation sample audio.")
