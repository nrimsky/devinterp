import torch as t
from tqdm import tqdm
import random
from model import MLP
from dataset import make_dataset, train_test_split
from dynamics import ablate_other_modes_fourier_basis, ablate_other_modes_embed_basis, get_magnitude_modes
from model_viz import viz_weights_modes, plot_mode_ablations, plot_magnitudes
from movie import run_movie_cmd
from dataclasses import dataclass, asdict
import json
from helpers import eval_model

@dataclass
class ExperimentParams:
  p: int = 53
  train_frac: float = 0.8
  hidden_size: int = 32
  lr: float = 0.01
  device = t.device('cuda' if t.cuda.is_available() else 'cpu')
  batch_size: int = 256
  embed_dim: int = 8
  tie_unembed: bool = True
  weight_decay:float = 0.0002
  movie: bool=True
  magnitude: bool=True
  ablation_fourier: bool=True
  ablation_embed: bool=False
  n_batches: int = 1000
  track_times: int = 20
  print_times: int = 10
  frame_times: int = 100
  freeze_middle: bool = False
  scale_linear_1_factor: int = 2
  save_activations: bool = False
  linear_1_tied: bool = False
  run_id: int = 0

  def save_to_file(self, fname):
    class_dict = asdict(self)
    with open(fname, "w") as f:
      json.dump(class_dict, f)

  @staticmethod
  def load_from_file(fname):
    with open(fname, "r") as f:
      class_dict = json.load(f)
    return ExperimentParams(**class_dict)

  def get_suffix(self):
    if self.run_id == 0:
      return f"P{self.p}_frac{self.train_frac}_hid{self.hidden_size}_emb{self.embed_dim}_tie{self.tie_unembed}_freeze{self.freeze_middle}"
    return f"P{self.p}_frac{self.train_frac}_hid{self.hidden_size}_emb{self.embed_dim}_tie{self.tie_unembed}_freeze{self.freeze_middle}_run{self.run_id}"

def test(model, dataset, device):
  n_correct = 0
  model.eval()
  with t.no_grad():
    for (x1, x2), y in dataset:
      out = model(x1.to(device), x2.to(device)).cpu()
      pred = t.argmax(out)
      if pred == y:
        n_correct += 1
  return n_correct / len(dataset)

def get_loss_only_modes(model, modes, test_dataset, params):
  model_copy = MLP(params)
  model_copy.to(params.device)
  model_copy.load_state_dict(model.state_dict())
  model_copy.eval()
  if params.ablation_fourier:
    model_copy.embedding.weight.data = ablate_other_modes_fourier_basis(model_copy.embedding.weight.detach().cpu(), modes, params.p).to(params.device)
  elif params.ablation_embed:
    model_copy.embedding.weight.data = ablate_other_modes_embed_basis(model_copy.embedding.weight.detach().cpu(), modes, params.p).to(params.device)
  return eval_model(model_copy, test_dataset, params.device).item()

def train(model, train_dataset, test_dataset, params):
  model = model.to(params.device)

  if params.freeze_middle:
    optimizer = t.optim.Adam(list(model.embedding.parameters()) + list(model.linear2.parameters()), weight_decay=params.weight_decay, lr=params.lr)
  else:
    optimizer = t.optim.Adam(model.parameters(), weight_decay=params.weight_decay, lr=params.lr)
  loss_fn = t.nn.CrossEntropyLoss()
  idx = list(range(len(train_dataset)))
  avg_loss = 0

  track_every = params.n_batches // params.track_times
  print_every = params.n_batches // params.print_times
  frame_every = params.n_batches // params.frame_times
  step = 0

  mode_loss_history = []
  magnitude_history = []

  for i in tqdm(range(params.n_batches)):
    with t.no_grad():
      model.eval()
      if i % print_every == 0:
        val_acc = test(model, test_dataset, params.device)
        avg_loss /= print_every
        print(f"Batch: {i} | Loss: {avg_loss} | Val Acc: {val_acc}")
        avg_loss = 0
      if i % track_every == 0:
        if params.magnitude:
          mags = get_magnitude_modes(model.embedding.weight.detach().cpu(), params.p)
          magnitude_history.append(mags)
        if params.ablation_fourier or params.ablation_embed:
          mode_losses = {}
          for mode in range(1, params.p//2 + 1):
            l_mode = get_loss_only_modes(model, [mode], test_dataset, params)
            mode_losses[mode] = l_mode
          mode_loss_history.append(mode_losses)
      if i % frame_every == 0 and params.movie:
        viz_weights_modes(model.embedding.weight.detach().cpu(), params.p, f"frames/embeddings_movie_{step:06}.png")  
        step += 1
    model.train()
    # Sample random batch of data
    batch_idx = random.sample(idx, params.batch_size)
    X_1 = t.stack([train_dataset[b][0][0] for b in batch_idx]).to(params.device)
    X_2 = t.stack([train_dataset[b][0][1] for b in batch_idx]).to(params.device)
    Y = t.stack([train_dataset[b][1] for b in batch_idx]).to(params.device)
    # Gradient update
    optimizer.zero_grad()
    out = model(X_1, X_2)
    loss = loss_fn(out, Y)
    avg_loss += loss.item()
    loss.backward()
    optimizer.step()
  val_acc = test(model, test_dataset, params.device)
  print(f"Final Val Acc: {val_acc}")
  return model, mode_loss_history, magnitude_history

if __name__ == "__main__":
    params = ExperimentParams(linear_1_tied=True, run_id=1, movie=False)
    params.save_to_file(f"models/params_{params.get_suffix()}.json")
    model = MLP(params)
    dataset = make_dataset(params.p)
    train_data, test_data = train_test_split(dataset, params.train_frac)
    model, mode_loss_history, magnitude_history = train(
        model=model,
        train_dataset=train_data,
        test_dataset=test_data,
        params=params
    )
    t.save(model.state_dict(), f"models/model_{params.get_suffix()}.pt")
    viz_weights_modes(model.embedding.weight.detach().cpu(), params.p, f"plots/final_embeddings_{params.get_suffix()}.png")  
    if len(mode_loss_history) > 0:
      plot_mode_ablations(mode_loss_history, f"plots/ablation_{params.get_suffix()}.png")
    if len(magnitude_history) > 0:
      plot_magnitudes(magnitude_history, params.p, f"plots/magnitudes_{params.get_suffix()}.png")
    if params.movie:
      run_movie_cmd(params.get_suffix())