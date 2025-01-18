import tiktoken

class BPE_Tokenizer():
    def __init__(self, encoding_name): # use o200k_base for the encoding_name
        self.encoding_name = encoding_name
        self.tokenzier = tiktoken.get_encoding(self.encoding_name)

    def tokenize(self, text):
        return self.tokenzier.encode(
            text=text,
            allowed_special={"<|endoftext|>"}
        )
    
    def detokenize(self, tokens):
        return self.tokenzier.decode(
            tokens=tokens,
            errors="strict" 
        )
    
    def get_vocab_size(self):
        return self.tokenzier.n_vocab
