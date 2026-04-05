import tiktoken

class Tokenizer:
    def __init__(self, encoding_name="gpt2"):
        """
        Initializes the standard GPT-2 BPE tokenizer.
        """
        self.encoder = tiktoken.get_encoding(encoding_name)
        self.eot_token = self.encoder._special_tokens['<|endoftext|>']
        self.vocab_size = self.encoder.n_vocab

    def encode(self, text):
        """
        Converts a raw string into a list of token integers.
        """
        return self.encoder.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, ids):
        """
        Converts a list of token integers back into a string.
        """
        return self.encoder.decode(ids)