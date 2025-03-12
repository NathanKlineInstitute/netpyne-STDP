import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt
import glob

sys.path.append(os.path.abspath(os.getcwd()))
sys.path.append(os.path.abspath(os.getcwd()) + "/neurosim/")
from conf import read_conf


def plot_performance(data):
    # plotting
    fig, ax = plt.subplots()
    # title = "EVOL vs EVOL+STDP-RL"
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
            
            if mean.sum() == 0:
                continue

            label = exp_name + ' '
            if k == 'Alpha':
                label = 'EVOL'
            elif k == 'Beta':
                label = 'EVOL+STDP-RL - During STDP'
            elif k == 'Gamma':
                label = 'EVOL+STDP-RL - Post-STDP'

            ax.fill_between(xAxies, 
                            np.clip(mean + std, a_min=0, a_max=500), 
                            np.clip(mean - std, a_min=0, a_max=500), 
                            alpha=0.2, linewidth=0)
            ax.plot(xAxies, mean, linewidth=2, label= label)

    ax.set_title(title)
    ax.set_xlabel("Generations")
    ax.set_ylabel("Performance")
    ax.legend(prop={'size': 9})
    
    plt.savefig(title + r".png", dpi=300)
    plt.close()


def main(configs, report, all):

    if all:
        files = (glob.glob(r'' + all + "/*.json"))
        configs_to_process = list()
        for f in files:
            for conf in configs:
                if f.split('/')[-1] != conf:
                    configs_to_process.append([conf, f])
    else:
      configs_to_process = [configs]

    for pair in configs_to_process:
        data = dict()
        for config in pair:
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
    parser.add_argument("--all", type=str, default="None")
    parser.add_argument("--report", type=str, default="None")

    args = parser.parse_args()
    args.configs = args.configs.replace("  ", " ").split(" ")
    
    if args.report == "None":
        args.report = "performance_verbos.csv"
        
    if args.all == "None":
        args.all = False
    
    if args.all and len(args.configs) > 1:
        print("All args activateed with more then one config")
    else:
        main(**vars(args))
