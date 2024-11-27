import json
import os

from datasets import load_dataset
#create localdir
local_dir="instrucion_files"
DATA_CACHE_DIR=os.path.join(os.path.dirname(__file__),local_dir)
os.makedirs(DATA_CACHE_DIR,exist_ok=True)
# load_dataset
ds = load_dataset("jayelm/natural-instructions")

num_proc = int(os.cpu_count()) // 2
# take out only definition inputs targets

train_dataset = ds["train"].map(
    lambda x: {
        "definition": x["definition"],
        "inputs": x["inputs"],
        "targets": x["targets"],
    },
    desc="processing",
    num_proc=num_proc,
)
val_dataset = ds["test"].map(
    lambda x: {
        "definition": x["definition"],
        "inputs": x["inputs"],
        "targets": x["targets"],
    },
    desc="processing",
    num_proc=num_proc,
)
# convert json format
train_instructions = [
    {"definition": definition, "inputs": inputs, "targets": targets}
    for definition, inputs, targets in zip(
        train_dataset["definition"], train_dataset["inputs"], train_dataset["targets"]
    )
]

val_intructions = [
    {"definition": definition, "inputs": inputs, "targets": targets}
    for definition, inputs, targets in zip(
        val_dataset["definition"], val_dataset["inputs"], val_dataset["targets"]
    )
]


from tqdm import tqdm
import tiktoken
import glob
#writ jason file
total_batches=10
filename="/instruction_finetuning"

batch_size=len(train_instructions)//total_batches
name=0
for batch_idx in tqdm(range(total_batches),desc=f"writing {filename}"):
    batch_idx=batch_idx*batch_size
    if batch_idx+batch_size<len(train_instructions):
        values=train_instructions[batch_idx:batch_idx+batch_size]
    else:
        values=train_instructions[batch_idx:] 
    file_path=os.path.join(DATA_CACHE_DIR,f"train_instruction{name}.json")
    with open(file_path,"w") as file:
        json.dump(values,file,ensure_ascii=False,indent=4)
    name+=1
    