import os
import urllib.request

import numpy as np
import tiktoken

file_path = "the_verdicct.text"
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

if not os.path.exists(url):
    with urllib.request.urlopen(url) as response:
        text_data = response.read().decode("utf-8")
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text_data)

else:
    with open(file_path, "r") as file:
        text_data = file.read()


enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens["<|endoftext|>"]
tokens = [eot]
tokens.extend(enc.encode(text_data))
tokens_np = np.array(tokens)


class DataloaderLite:
    def __init__(self, corpus, B, T, process_rank, num_process):
        self.B = B
        self.T = T
        self.proces_rank = process_rank
        self.num_process = num_process
        self.current_position = self.B * self.T * process_rank
        self.corpus = corpus

    def get_batch(self):
        buf = self.corpus[
            self.current_position : self.current_position + self.B * self.T + 1
        ]
        x = buf[:-1].view(self.B, self.T)
        y = buf[1:].view(self.B, self.T)
        self.current_position += self.B * self.T * self.num_process
        return x, y
