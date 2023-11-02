import json
import matplotlib.pyplot as plt
from glob import glob
from collections import defaultdict


def extract_slopes_from_json(json_files):
    slopes = []
    n_mults = []
    for json_file in json_files:
        with open(json_file, "r") as f:
            results = json.load(f)

        avg_over_runs = defaultdict(list)
        for p in results.keys():
            for run_id in results[p].keys():
                avg_over_runs[p].append(results[p][run_id][0])
        for p in avg_over_runs.keys():
            avg_over_runs[p] = sum(avg_over_runs[p]) / len(avg_over_runs[p])
        p1 = list(avg_over_runs.keys())[0]
        p2 = list(avg_over_runs.keys())[-1]
        avg_lambda_p1 = float(avg_over_runs[p1])
        avg_lambda_p2 = float(avg_over_runs[p2])
        slope = abs((avg_lambda_p2 - avg_lambda_p1) / (float(p2) - float(p1)))
        slopes.append(slope)

        # Extract n_multiplier from the file name
        n_mult = int(json_file.split("_")[-1].split(".")[0])
        n_mults.append(n_mult)

    return n_mults, slopes

def extract_val_from_json(json_files, p_1 = True):
    vals = []
    n_mults = []
    for json_file in json_files:
        with open(json_file, "r") as f:
            results = json.load(f)

        avg_over_runs = defaultdict(list)
        for p in results.keys():
            for run_id in results[p].keys():
                avg_over_runs[p].append(results[p][run_id][0])
        for p in avg_over_runs.keys():
            avg_over_runs[p] = sum(avg_over_runs[p]) / len(avg_over_runs[p])
        p_val = list(avg_over_runs.keys())[0 if p_1 else -1]
        avg_lambda = float(avg_over_runs[p_val])
        vals.append(avg_lambda)
        # Extract n_multiplier from the file name
        n_mult = int(json_file.split("_")[-1].split(".")[0])
        n_mults.append(n_mult)
    return n_mults, vals

def reproduce_plot_from_json(extract_fn = extract_slopes_from_json, title="Slope of $\lambda$ vs p vs. n_multiplier", yaxis = "Slope of $\lambda$ vs p", filename="lambda_vs_p.png"):
    json_files = glob(f"results_n_mult_*.json")
    n_mults, slopes = extract_fn(json_files)

    # Sort the data by n_mults to ensure the plot is correct
    sorted_data = sorted(zip(n_mults, slopes))
    n_mults_sorted, slopes_sorted = zip(*sorted_data)
    n_mults_sorted = n_mults_sorted[:-4]
    slopes_sorted = slopes_sorted[:-4]

    plt.clf()
    plt.figure()
    # plt.xscale("log")
    # Optionally, set axis limits if needed
    # plt.xlim([min(n_mults_sorted), max(n_mults_sorted)])
    # plt.ylim([min(slopes_sorted), max(slopes_sorted)])

    plt.plot([i / 3 for i in range(-6, 7)][:-4], slopes_sorted, marker="o")
    # make x ticks [10**(i/3) for i in range(-6, 7)] but displayed to 2 sig figs
    plt.xticks([i / 3 for i in range(-6, 7)][:-4], [f"{10**(i/3):.2g}" for i in range(-6, 7)][:-4])

    plt.title(title)
    plt.xlabel("n_multiplier")
    plt.ylabel(yaxis)
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    reproduce_plot_from_json(filename="slope_vs_nmult.png")
    reproduce_plot_from_json(extract_fn=lambda files: extract_val_from_json(files, p_1=False), title="p=17 $\lambda$ vs n_multiplier", yaxis="$\lambda$", filename="lambda_vs_nmult_p17.png")
    reproduce_plot_from_json(extract_fn=lambda files: extract_val_from_json(files, p_1=True), title="p=41 $\lambda$ vs n_multiplier", yaxis="$\lambda$", filename="lambda_vs_nmult_p41.png")
