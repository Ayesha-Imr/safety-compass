from types import SimpleNamespace

import torch

from safety_compass.concept import ConceptDirectionExtractor


class FakeTokenizer:
    pad_token = "<pad>"
    eos_token = "</s>"
    pad_token_id = 0
    eos_token_id = 0
    padding_side = "right"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        return messages[0]["content"]

    def __call__(self, prompts, return_tensors, padding, truncation, max_length):
        tokenized = {
            "short": [10, 11],
            "long": [20, 21, 22, 23],
        }
        max_len = max(len(tokenized[prompt]) for prompt in prompts)
        input_ids = []
        attention_mask = []
        for prompt in prompts:
            ids = tokenized[prompt]
            pad_len = max_len - len(ids)
            if self.padding_side == "left":
                input_ids.append([0] * pad_len + ids)
                attention_mask.append([0] * pad_len + [1] * len(ids))
            else:
                input_ids.append(ids + [0] * pad_len)
                attention_mask.append([1] * len(ids) + [0] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
        }


class FakeModel:
    device = torch.device("cpu")

    def __call__(self, input_ids, attention_mask, output_hidden_states):
        hidden = input_ids.float().unsqueeze(-1)
        return SimpleNamespace(hidden_states=(hidden, hidden + 100))


def test_extract_hidden_states_uses_last_real_token_with_left_padding():
    extractor = ConceptDirectionExtractor(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        concept_config={"name": "test"},
        model_config={"num_layers": 1, "hidden_dim": 1, "extraction_batch_size": 2},
    )

    activations = extractor._extract_hidden_states([
        [{"role": "user", "content": "short"}],
        [{"role": "user", "content": "long"}],
    ])

    assert activations.shape == (2, 2, 1)
    assert activations[0, 0, 0] == 11
    assert activations[1, 0, 0] == 23
