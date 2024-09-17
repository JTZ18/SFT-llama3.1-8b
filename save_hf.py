import os
import fire
import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

def load_and_push_to_hub(
    model_name_or_path: str = "./weights/full",  # Path to the local model directory
    hub_model_id: str = "jtz18/llama3-8b-jon",  # Your Hugging Face model ID
    push_to_hub: bool = True,  # Flag to push to Hugging Face Hub
    hf_token: str = None,  # Hugging Face token
):
    if hf_token:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
        os.environ["HF_TOKEN"] = hf_token

    # Load the model with ignore_mismatched_sizes=True
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=torch.bfloat16, ignore_mismatched_sizes=True
    )
    model.config.use_cache = False

    # Load the tokenizer
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, padding_side="right", use_fast=False
    )

    # Push to Hugging Face Hub
    if push_to_hub:
        try:
            model.push_to_hub(hub_model_id)
            tokenizer.push_to_hub(hub_model_id)
            logger.info(f"Successfully pushed model and tokenizer to Hugging Face Hub: {hub_model_id}")
        except Exception as e:
            logger.error(f"Failed to push to hub: {e}")

if __name__ == "__main__":
    fire.Fire(load_and_push_to_hub)