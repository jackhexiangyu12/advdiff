import pandas as pd

df = pd.read_parquet("hf://datasets/lmms-lab/Video-MME/videomme/test-00000-of-00001.parquet")

print(df)

print(df['train'][0])