import json
from typing import Any, Dict, List

import torch
from loguru import logger
from torch.utils.data import Dataset


class SFTDataset(Dataset):
    def __init__(self, file, tokenizer, max_seq_length, template):
        self.tokenizer = tokenizer
        self.system_format = template["system_format"]
        self.user_format = template["user_format"]
        self.assistant_format = template["assistant_format"]

        self.max_seq_length = max_seq_length
        logger.info("Loading data: {}".format(file))
        with open(file, "r", encoding="utf8") as f:
            data_list = f.readlines()
        logger.info("There are {} data in dataset".format(len(data_list)))
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        data = json.loads(data)
        input_ids, target_mask = [], []

        # setting system information
        if self.system_format is not None:
            system = data["system"].strip() if "system" in data.keys() else None

            if system is not None:
                system_text = self.system_format.format(content=system)
                input_ids = self.tokenizer.encode(system_text, add_special_tokens=False)
                target_mask = [0] * len(input_ids)

        conversations = data["conversations"]

        valid = True
        for i in range(0, len(conversations) - 1, 2):
            if (
                conversations[i]["role"] != "user"
                or conversations[i + 1]["role"] != "assistant"
            ):
                logger.error(f"Incorrect role order at index {index}: {conversations}")
                valid = False
                break

            human = conversations[i]["content"].strip()
            assistant = conversations[i + 1]["content"].strip()

            human = self.user_format.format(
                content=human, stop_token=self.tokenizer.eos_token
            )
            assistant = self.assistant_format.format(
                content=assistant, stop_token=self.tokenizer.eos_token
            )

            input_tokens = self.tokenizer.encode(human, add_special_tokens=False)
            output_tokens = self.tokenizer.encode(assistant, add_special_tokens=False)

            input_ids += input_tokens + output_tokens
            target_mask += [0] * len(input_tokens) + [1] * len(output_tokens)

        if valid:
            assert len(input_ids) == len(target_mask)

            input_ids = input_ids[: self.max_seq_length]
            target_mask = target_mask[: self.max_seq_length]
            attention_mask = [1] * len(input_ids)
            assert len(input_ids) == len(target_mask) == len(attention_mask)
            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "target_mask": target_mask,
            }
            return inputs
        else:
            # Skip this example and fetch the next one
            new_index = (index + 1) % len(self.data_list)
            return self.__getitem__(new_index)


class SFTDataCollator(object):
    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Find the maximum length in the batch
        lengths = [len(x["input_ids"]) for x in batch if x["input_ids"] is not None]
        # Take the maximum length in the batch, if it exceeds max_seq_length, take max_seq_length
        batch_max_len = min(max(lengths), self.max_seq_length)

        input_ids_batch, attention_mask_batch, target_mask_batch = [], [], []
        # Truncate and pad
        for x in batch:
            input_ids = x["input_ids"]
            attention_mask = x["attention_mask"]
            target_mask = x["target_mask"]
            if input_ids is None:
                logger.info("some input_ids is None")
                continue
            padding_len = batch_max_len - len(input_ids)
            # Pad
            input_ids = input_ids + [self.pad_token_id] * padding_len
            attention_mask = attention_mask + [0] * padding_len
            target_mask = target_mask + [0] * padding_len
            # Truncate
            input_ids = input_ids[: self.max_seq_length]
            attention_mask = attention_mask[: self.max_seq_length]
            target_mask = target_mask[: self.max_seq_length]

            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            target_mask_batch.append(target_mask)

        # Convert lists to tensors to get the final model input
        input_ids_batch = torch.tensor(input_ids_batch, dtype=torch.long)
        attention_mask_batch = torch.tensor(attention_mask_batch, dtype=torch.long)
        target_mask_batch = torch.tensor(target_mask_batch, dtype=torch.long)

        labels = torch.where(target_mask_batch == 1, input_ids_batch, -100)
        inputs = {
            "input_ids": input_ids_batch,
            "attention_mask": attention_mask_batch,
            "labels": labels,
        }
        return inputs

# 在开始训练之前，确保 `model_id` 在支持的模型列表中
supported_model_ids = ["model1", "model2", "google/gemma-2-9b-it"]  # 示例支持的模型列表

def train_lora(model_id):
    if model_id not in supported_model_ids:
        raise AssertionError(f"model_id {model_id} not supported")
    # 训练逻辑
    ...

if __name__ == "__main__":
    model_id = "google/gemma-2-9b-it"
    try:
        train_lora(model_id)
    except AssertionError as e:
        logger.error(e)
        logger.info("Proceed to the next model...")
