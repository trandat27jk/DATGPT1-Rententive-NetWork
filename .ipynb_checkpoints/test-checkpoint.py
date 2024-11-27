import tiktoken
import torch
import torch.nn as nn

from RetNet import RetNet, RetnetConfig

checkpoint = torch.load("model_04000.pt")
model = RetNet(RetnetConfig)
model.load_state_dict(checkpoint["model"])


inputs = "hello how are you"
enc = tiktoken.get_encoding("gpt2")
inputs = enc.encode(inputs)
inputs = torch.tensor(inputs).unsqueeze(0)
outputs = model.generate(inputs, max_new_tokens=100, top_k=10)
outputs = outputs.squeeze(0)
print(enc.decode(outputs.tolist()))
