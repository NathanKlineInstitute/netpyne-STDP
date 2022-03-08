from cProfile import label
import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.getcwd()))
sys.path.append(os.path.abspath(os.getcwd()) + "/neurosim/")
from conf import read_conf


def plot_performance(data):
    # plotting
    fig, ax = plt.subplots()
    title = ""
    for exp_name in data.keys():
        if title == "":
            title = exp_name
        else:
            title += " VS " + exp_name

        feilds = list(data[exp_name].keys())
        
        for k, v in data[exp_name].items():
            if k == feilds[-1]:
                continue
            
            mean = np.array(data[exp_name][k]["data"]).mean(1)
            std = np.array(data[exp_name][k]["data"]).std(1)
            gen = len(data[exp_name][k]["data"])
            xAxies = np.linspace(0, gen, gen)

            ax.fill_between(xAxies, std + mean, std - mean, alpha=0.2, linewidth=0)
            ax.plot(xAxies, mean, linewidth=2, label= exp_name + ' ' + k)

    ax.set_title(title)
    ax.set_xlabel("Generations")
    ax.set_ylabel("Performance")
    ax.legend(prop={'size': 6})
    
    plt.savefig(title + r".png", dpi=300)


def main(configs, report):

    data = dict()
    configs = configs.replace("  ", " ").split(" ")
    for config in configs:
        configs_jason = read_conf(config)
        sim_name = configs_jason["sim"]["outdir"].split("/")[-1]
        data[sim_name] = dict()
        feilds = list()
        n_generations = 0
        open_file = configs_jason["sim"]["outdir"] + "/" + report
        with open(open_file, "rt") as f:
            # read lagent
            st = f.readline()
            for field in st.split(","):
                field = field.replace("\n", "")
                feilds.append(field)
                data[sim_name][field] = {"data": list(), "feild_size": 0}
            # read feild size
            st = f.readline().split(",")
            for i, field in enumerate(data[sim_name].keys()):
                data[sim_name][field]["feild_size"] = (
                    int(st[i]) if int(st[i]) > 0 else 1
                )

            # space
            st = f.readline().split(",")

            # data
            for i_d, d in enumerate(f):
                n_generations += 1
                for i_v, v in enumerate(
                    d.replace("]\n", "").replace(" ", "").split("]")
                ):
                    data[sim_name][feilds[i_v]]["data"].append(
                        np.array(v.split(","), dtype=float)
                    )
    plot_performance(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=str)
    parser.add_argument("--report", type=str, default="None")

    args = parser.parse_args()
    if args.report == "None":
        args.report = "performance_verbos.csv"

    main(**vars(args))
