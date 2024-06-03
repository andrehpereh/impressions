import tiktoken

class TextTokenizer:
    def __init__(self, tokenizer_type, text_data=None, stoi=None, itos=None, special_token="<UNK>"):
        self.text_data = text_data
        self.stoi = stoi
        self.itos = itos
        self.special_token = special_token
        self.tokenizer_type = tokenizer_type
        print("The tokenizer type is :", tokenizer_type)
        self._initialize_tokenizer()

    def _initialize_tokenizer(self):
        if self.tokenizer_type == "char":
            self._initialize_char_tokenizer()
        elif self.tokenizer_type == "gpt-4":
            self._initialize_word_tokenizer()
        else:
            raise ValueError("Unsupported tokenizer type. Choose 'char' or 'gpt-4'.")

    def _initialize_char_tokenizer(self):
        if self.stoi is not None and self.itos is not None:
            self.encode_func = lambda s: [self.stoi.get(c, self.stoi[self.special_token]) for c in s]
            self.decode_func = lambda l: ''.join([self.itos.get(i, self.special_token) for i in l])
        else:
            chars = sorted(list(set(self.text_data)))
            if self.special_token not in chars:
                chars.append(self.special_token)
            self.stoi = {ch: i for i, ch in enumerate(chars)}
            self.itos = {i: ch for i, ch in enumerate(chars)}
            self.encode_func = lambda s: [self.stoi.get(c, self.stoi[self.special_token]) for c in s]
            self.decode_func = lambda l: ''.join([self.itos.get(i, self.special_token) for i in l])

    def _initialize_word_tokenizer(self):
        encoding = tiktoken.get_encoding("cl100k_base")
        enc = tiktoken.encoding_for_model("gpt-4")
        self.special_token_idx = list(set(enc.encode(self.special_token)))
        if self.stoi is None and self.itos is None:
            self.text_token_idx= list(set(enc.encode(self.text_data)))
            print("special_token_idx", self.special_token_idx)
            self.unique_token_idx = sorted(self.text_token_idx+ self.special_token_idx)
            self.stoi = {num: i for i, num in enumerate(self.unique_token_idx)}
            self.itos = {i: num for num, i in self.stoi.items()}

        def encode(text):
            encoded_numbers = enc.encode(text)
            return [self.stoi.get(num, self.special_token_idx[0]) for num in encoded_numbers] # This should be improved, since right now it only retrieves the last token_id of the special token.

        def decode(encoded_numbers):
            decoded_sequence = [self.itos.get(num, self.special_token) for num in encoded_numbers]
            return enc.decode(decoded_sequence)

        self.encode_func = encode
        self.decode_func = decode

    def encode(self, text):
        return self.encode_func(text)

    def decode(self, encoded_numbers):
        return self.decode_func(encoded_numbers)

    def get_stoi(self):
        return self.stoi

    def get_itos(self):
        return self.itos

