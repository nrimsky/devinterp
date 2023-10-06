import torch as t
import random

def get_all_pairs(p):
  pairs = []
  for i in range(p):
    for j in range(p):
      pairs.append((i,j))
  return set(pairs)

def make_dataset(p):
  data = []
  pairs = get_all_pairs(p)
  for a, b in pairs:
    data.append(((t.tensor(a), t.tensor(b)), t.tensor((a + b) % p)))
  return data

def train_test_split(dataset, train_split_proportion):
  l = len(dataset)
  train_len = int(train_split_proportion * l)
  idx = list(range(l))
  random.shuffle(idx)
  train_idx = idx[:train_len]
  test_idx = idx[train_len:]
  return [dataset[i] for i in train_idx], [dataset[i] for i in test_idx]