from tokenizers import Tokenizer


class basic_sampling_fn:
    def __init__(self, tokenizer: Tokenizer, ctx_length: int, key="text") -> None:
        self.tokenizer = tokenizer
        self.ctx_length = ctx_length
        self.key = key

    def __call__(self, element):
        outputs = self.tokenizer(
            element[self.key],
            truncation=True,
            max_length=self.ctx_length,
            return_overflowing_tokens=True,
            return_length=True,
        )
        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            if length == self.ctx_length:
                input_batch.append(input_ids)
        return {"input_ids": input_batch}


class overlapped_sampling_fn:
    def __init__(self, tokenizer: Tokenizer, ctx_length: int, key="text") -> None:
        self.tokenizer = tokenizer
        self.ctx_length = ctx_length
        self.key = key

    def __call__(self, element):
        outputs = self.tokenizer(
            element[self.key],
            truncation=True,
            max_length=self.ctx_length,
            return_overflowing_tokens=True,
            return_length=True,
        )
        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            if length != self.ctx_length:
                continue
            if len(input_batch) > 0:
                # create overlapped sample
                tail_of_prev_sample = input_batch[-1][self.ctx_length // 2 :]
                head_of_current_sample = input_ids[: self.ctx_length // 2]
                input_batch.append(tail_of_prev_sample + head_of_current_sample)
            input_batch.append(input_ids)
        return {"input_ids": input_batch}
