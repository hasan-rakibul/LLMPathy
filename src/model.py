from transformers import AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType

def get_model(config):
    if 'roberta' in config.checkpoint:
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=config.checkpoint,
            num_labels=1
        )
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=2,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none"
        )
    elif 'mistral' in config.checkpoint:
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=config.checkpoint,
            num_labels=1,
            device_map="auto"
        )
        model.config.pad_token_id = model.config.eos_token_id
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=2,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            target_modules=[
                "q_proj",
                "v_proj"
            ]
        )
    elif 'llama' in config.checkpoint:
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=config.checkpoint,
            num_labels=1,
            device_map="auto",
            offload_folder="offload",
            trust_remote_code=True
        )
        model.config.pad_token_id = model.config.eos_token_id
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=16,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            target_modules=[
                "q_proj",
                "v_proj"
            ]
        )
  
    model = get_peft_model(model, peft_config)
    return model