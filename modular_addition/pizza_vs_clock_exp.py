import torch as t
from train import ExperimentParams
from model import MLP
from model_viz import plot_ft_input_output_activations

def experiment(params, a, model_fname):
    params = ExperimentParams()
    params.save_activations = True
    model = MLP(params)
    model.load_state_dict(t.load(model_fname))
    for i in range(params.p):
        model(t.tensor([a]), t.tensor([i]))
    plot_ft_input_output_activations(model.saved_activations, a, params.p, f"plots/ft_input_output_activations_{params.get_suffix()}_a{a}.png")

if __name__ == "__main__":
    params = ExperimentParams.load_from_file("models/params_P53_frac0.8_hid32_emb8_tieTrue_freezeFalse.json")
    experiment(params, 17, f"models/model_{params.get_suffix()}.pt")


