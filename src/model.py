from transformers import AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType
from torch import nn

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

class SSLRoberta(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = AutoModelForSequenceClassification.from_pretrained(
            config.checkpoint,
            num_labels=768
        )

        self.fc_m = nn.Linear(768, 1)
        self.fc_v = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        x_feat_m = nn.functional.dropout(outputs.logits, p=self.config.train.dropout, training=True)
        x_feat_v = nn.functional.dropout(outputs.logits, p=self.config.train.dropout, training=True)

        m = self.fc_m(x_feat_m)
        v = self.fc_v(x_feat_v)

        return m, v