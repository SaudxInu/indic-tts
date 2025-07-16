# * Install unsloth, PEFT, Weights & Biases, SNAC, pandas, soundfile and loguru.
# pip install unsloth peft==0.15.2 wandb snac pandas soundfile loguru

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
STAGE = 1
HUGGINGFACE_USERNAME = ""  # ! Fill.
if STAGE == 1:
    # * You need to request access to the model at https://huggingface.co/canopylabs/3b-hi-pretrain-research_release.
    BASE_MODEL = "canopylabs/3b-hi-pretrain-research_release"
    TARGET_MODULES = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "down_proj",
        "gate_proj",
        "lm_head",
        "embed_tokens",
    ]
    TRAIN_CSV_PATH = "data/data_stage_1.csv"
    VALID_CSV_PATH = "data/data_eval_stage_2.csv"
    LR = 2e-4
    EPOCHS = 2
    MODEL_NAME = f"indic-tts-lora-stage-{STAGE}"
else:
    BASE_MODEL = f"{HUGGINGFACE_USERNAME}/indic-tts-lora-stage-1"
    TARGET_MODULES = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "down_proj",
        "gate_proj",
    ]
    TRAIN_CSV_PATH = "data/data_train_stage_2.csv"
    VALID_CSV_PATH = "data/data_eval_stage_2.csv"
    LR = 2e-4
    EPOCHS = 2
    MODEL_NAME = f"indic-tts-lora-tamil-stage-{STAGE}"
TRAIN_NUM_SAMPLES = None
EVAL_NUM_SAMPLES = 250
MAX_SEQ_LENGTH = 2048
PER_DEVICE_TRAIN_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4
WANDB_USERNAME = ""  # ! Fill.
WANDB_PROJECT = MODEL_NAME
WANDB_LOG_MODEL = "checkpoint"
WANDB_RUN_NAME = f"{MODEL_NAME}-training"
WANDB_RUN_ID = None
SEED = 3407
HUGGINGFACE_TOKEN = ""  # ! Fill.
WANDB_TOKEN = ""  # ! Fill.

# * Use the following command to start the training: python train_orpheus.py

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
    load_in_4bit=True,
    max_seq_length=MAX_SEQ_LENGTH,
    token=HUGGINGFACE_TOKEN,
)
logger.success(f"Loaded model: {BASE_MODEL}")

# Load the special tokens for the tokenizer.
tokeniser_length = 128256

start_of_text_id = 128000
end_of_text_id = 128009
start_of_speech_id = tokeniser_length + 1
end_of_speech_id = tokeniser_length + 2
start_of_human_id = tokeniser_length + 3
end_of_human_id = tokeniser_length + 4
start_of_ai_id = tokeniser_length + 5
end_of_ai_id = tokeniser_length + 6
pad_token_id = tokeniser_length + 7
audio_start_id = tokeniser_length + 10

start_of_text_token = tokenizer.decode([start_of_text_id])
end_of_text_token = tokenizer.decode([end_of_text_id])
start_of_speech_token = tokenizer.decode([start_of_speech_id])
end_of_speech_token = tokenizer.decode([end_of_speech_id])
start_of_human_token = tokenizer.decode([start_of_human_id])
end_of_human_token = tokenizer.decode([end_of_human_id])
start_of_ai_token = tokenizer.decode([start_of_ai_id])
end_of_ai_token = tokenizer.decode([end_of_ai_id])
pad_token = tokenizer.decode([pad_token_id])
audio_start_token = tokenizer.decode([audio_start_id])

logger.success("Load special tokens for the tokenizer.")

# Set the padding token and padding side.
tokenizer.pad_token = pad_token
tokenizer.padding_side = "left"
logger.success("Set padding token and padding side for the tokenizer.")

# Get parameter efficient fine-tuning model.
model = FastLanguageModel.get_peft_model(
    model,
    r=192,
    target_modules=TARGET_MODULES,
    lora_alpha=384,
    random_state=SEED,
)
logger.success("Initialized parameter efficient fine-tuning model.")

# Load training and validation datasets.
# The dataset should be in CSV format with columns user (str), language (str), utterance (str), and snac_codes (list of lists).
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


# Flatten and get SNAC token IDs from the audio codes.
def flatten_and_get_audio_input_ids(row):
    audio_codes = row["snac_codes"]
    if isinstance(audio_codes, str):
        audio_codes = eval(audio_codes)
    snac_token_ids = []
    for i in range(len(audio_codes[0])):
        snac_token_ids.append(audio_codes[0][i] + 128266)
        snac_token_ids.append(audio_codes[1][2 * i] + 128266 + 4096)
        snac_token_ids.append(audio_codes[2][4 * i] + 128266 + (2 * 4096))
        snac_token_ids.append(audio_codes[2][(4 * i) + 1] + 128266 + (3 * 4096))
        snac_token_ids.append(audio_codes[1][(2 * i) + 1] + 128266 + (4 * 4096))
        snac_token_ids.append(audio_codes[2][(4 * i) + 2] + 128266 + (5 * 4096))
        snac_token_ids.append(audio_codes[2][(4 * i) + 3] + 128266 + (6 * 4096))
    row["snac_token_ids"] = snac_token_ids
    return row


train_dataset = train_dataset.map(flatten_and_get_audio_input_ids)
eval_dataset = eval_dataset.map(flatten_and_get_audio_input_ids)
logger.success("Flattened and extracted SNAC token IDs from audio codes.")

# Filter out rows with empty or None audio codes.
train_dataset = train_dataset.filter(
    lambda x: x["snac_token_ids"] is not None and len(x["snac_token_ids"]) > 0
)
eval_dataset = eval_dataset.filter(
    lambda x: x["snac_token_ids"] is not None and len(x["snac_token_ids"]) > 0
)
logger.success("Filtered datasets to remove rows with empty or None audio codes.")


# Remove duplicate frames from the audio codes.
def remove_duplicate_frames(row):
    vals = row["snac_token_ids"]
    if len(vals) % 7 != 0:
        raise ValueError("Input list length must be divisible by 7")
    result = vals[:7]
    for i in range(7, len(vals), 7):
        current_first = vals[i]
        previous_first = result[-7]
        if current_first != previous_first:
            result.extend(vals[i : i + 7])
    row["snac_token_ids"] = result
    return row


train_dataset = train_dataset.map(remove_duplicate_frames)
eval_dataset = eval_dataset.map(remove_duplicate_frames)
logger.success("Removed duplicate frames from audio codes.")


# Define a function to format the prompt for each row in the dataset.
def format_text(row):
    text = (
        f"{start_of_human_token}{start_of_text_token}{row['language']}{row['user']}: {row['utterance']}{end_of_text_token}"
        f"{end_of_human_token}{start_of_ai_token}{start_of_speech_token}"
        f"{tokenizer.decode(row['snac_token_ids'])}{end_of_speech_token}{end_of_ai_token}"
    )
    eval_text_user = (
        f"{start_of_human_token}{start_of_text_token}{row['language']}{row['user']}: {row['utterance']}{end_of_text_token}"
        f"{end_of_human_token}{start_of_ai_token}{start_of_speech_token}"
    )
    eval_text_no_user = (
        f"{start_of_human_token}{start_of_text_token}{row['utterance']}{end_of_text_token}"
        f"{end_of_human_token}{start_of_ai_token}{start_of_speech_token}"
    )
    row["text"] = text
    row["eval_text_user"] = eval_text_user
    row["eval_text_no_user"] = eval_text_no_user
    return row


train_dataset = train_dataset.map(format_text)
eval_dataset = eval_dataset.map(format_text)
logger.success("Formatted text for training and evaluation datasets.")


# Tokenize the text in the datasets without adding special tokens.
def tokenize_function(example):
    return tokenizer(
        example["text"],
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    )


train_dataset = train_dataset.map(tokenize_function)
eval_dataset = eval_dataset.map(tokenize_function)
logger.success("Tokenized text in the datasets without adding special tokens.")

# Set training arguments.
training_args = SFTConfig(
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    optim="adamw_8bit",
    learning_rate=LR,
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

# Inference with the trained model.
FastLanguageModel.for_inference(model)
logger.success(f"Model {MODEL_NAME} is ready for inference.")

# Load the SNAC model for audio decoding.
snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
logger.success("Loaded SNAC model for audio decoding.")


# Function to generate audio from a dataset row.
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


# Generate and save some examples.
train_sample = generate_audio(train_dataset[0], model, tokenizer, True)
if train_sample is None:
    logger.error("Failed to generate audio for training sample.")
else:
    sf.write(f"train_{STAGE}.wav", train_sample, 24000)
    logger.success("Generated and saved training sample audio.")


dir_ = f"eval_{STAGE}/"
os.makedirs(dir_, exist_ok=True)
for i in range(10):
    eval_sample = generate_audio(eval_dataset[i], model, tokenizer, True)
    if eval_sample is None:
        logger.error(f"Failed to generate audio for evaluation sample {i}.")
    else:
        filename = dir_ + f"eval_{i}.wav"
        sf.write(filename, eval_sample, 24000)
        logger.success(f"Generated and saved evaluation sample audio as {filename}.")

# Save the model and tokenizer.
model.save_pretrained_merged(
    f"{HUGGINGFACE_USERNAME}/{MODEL_NAME}",
    tokenizer,
    save_method="merged_16bit",
)
logger.success("Saved the model and tokenizer locally.")

model.push_to_hub_merged(
    f"{HUGGINGFACE_USERNAME}/{MODEL_NAME}",
    tokenizer,
    save_method="merged_16bit",
    token=HUGGINGFACE_TOKEN,
)
logger.success("Pushed the model and tokenizer to the Hugging Face Hub.")
