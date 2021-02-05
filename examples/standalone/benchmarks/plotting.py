#!/usr/bin/env python3

import json
import os
from argparse import ArgumentParser
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt


if __name__ == "__main__":
    usage = "usage: python %(prog)s <data_dir>"
    parser = ArgumentParser(usage=usage)
    parser.add_argument(
        "data_dir",
        type=str,
        action="store",
        help="directory containing potential subdirectories with "
        "the json files with performance data",
    )
    args = parser.parse_args()

    # setup what to plot
    fontsize = 10
    markersize = 4
    plot_variance = False
    plots = {
        "per_timestep": {
            "timers": ["mainloop", "DynCore", "Remapping", "TracerAdvection"],
            "title": "Performance history of components",
            "x_axis_label": "Date of benchmark",
            "y_axis_label": "Execution time per timestep [s]",
        },
        "absolute_time": {
            "timers": ["total", "initialization"],
            "title": "Performance history of absolute runtime",
            "x_axis_label": "Date of benchmark",
            "y_axis_label": "Execution time [s]",
        },
    }
    linestyles = {
        "mainloop": "-o",
        "DynCore": "--o",
        "Remapping": "-.o",
        "TracerAdvection": ":o",
        "total": "-o",
        "initialization": "--o",
    }
    backends = {
        "python/gtcuda": {"short_name": "gtcuda", "color": "#d62728"},
        "python/gtx86": {"short_name": "gtx86", "color": "#2ca02c"},
        "python/numpy": {"short_name": "numpy", "color": "#1f77b4"},
        "fortran": {"short_name": "f90", "color": "#7f7f7f"},
    }
    filter = "c128"

    # collect and sort the data
    alldata = []
    for subdir, dirs, files in os.walk(args.data_dir):
        for file in files:
            fullpath = os.path.join(subdir, file)
            if fullpath.endswith(".json"):
                with open(fullpath) as f:
                    data = json.load(f)
                    if filter in data["setup"]["dataset"]:
                        alldata.append(data)
    alldata.sort(
        key=lambda k: datetime.strptime(k["setup"]["timestamp"], "%d/%m/%Y %H:%M:%S")
    )

    for plottype, plot_config in plots.items():
        matplotlib.rcParams.update({"font.size": fontsize})
        plt.figure()
        for backend, backend_config in backends.items():
            specific = [x for x in alldata if x["setup"]["version"] == backend]
            if specific:
                for timer in plot_config["timers"]:
                    plt.plot(
                        [
                            datetime.strptime(
                                elememt["setup"]["timestamp"], "%d/%m/%Y %H:%M:%S"
                            )
                            for elememt in specific
                        ],
                        [
                            elememt["times"][timer]["mean"]
                            / (
                                (elememt["setup"]["timesteps"] - 1)
                                if plottype == "per_timestep"
                                else 1
                            )
                            for elememt in specific
                        ],
                        linestyles[timer],
                        markersize=markersize,
                        label=backend_config["short_name"] + " " + timer,
                        color=backend_config["color"],
                    )
                    if plot_variance:
                        plt.fill_between(
                            [
                                datetime.strptime(
                                    elememt["setup"]["timestamp"], "%d/%m/%Y %H:%M:%S"
                                )
                                for elememt in specific
                            ],
                            [
                                elememt["times"][timer]["maximum"]
                                / (
                                    (elememt["setup"]["timesteps"] - 1)
                                    if plottype == "per_timestep"
                                    else 1
                                )
                                for elememt in specific
                            ],
                            [
                                elememt["times"][timer]["minimum"]
                                / (
                                    (elememt["setup"]["timesteps"] - 1)
                                    if plottype == "per_timestep"
                                    else 1
                                )
                                for elememt in specific
                            ],
                            color=backend_config["color"],
                            alpha=0.3,
                        )

        ax = plt.gca()
        plt.gcf().autofmt_xdate(rotation=45, ha="right")
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m/%d"))
        plt.xticks(fontsize=fontsize)
        plt.ylabel(plot_config["y_axis_label"])
        plt.xlabel(plot_config["x_axis_label"])
        plt.yticks(fontsize=fontsize)
        plt.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.05),
            ncol=2,
            fancybox=True,
            shadow=True,
            fontsize=fontsize * 0.8,
            handlelength=5,
        )
        plt.title(plot_config["title"], pad=20)
        plt.figtext(
            0.5,
            0.01,
            "data: "
            + alldata[0]["setup"]["dataset"]
            + "   timesteps:"
            + str(alldata[0]["setup"]["timesteps"]),
            wrap=True,
            horizontalalignment="center",
            fontsize=fontsize,
        )
        ax.set_facecolor("white")
        plt.grid(color="silver", alpha=0.4)
        plt.gcf().set_size_inches(8, 6)
        plt.savefig("history_" + plottype + ".png", dpi=100)
