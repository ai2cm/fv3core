import json
import os
from argparse import ArgumentParser
from datetime import datetime

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

    # collect and sort the data
    alldata = []
    for subdir, dirs, files in os.walk(args.data_dir):
        for file in files:
            fullpath = os.path.join(subdir, file)
            if fullpath.endswith(".json"):
                with open(fullpath) as f:
                    alldata.append(json.load(f))
    alldata.sort(
        key=lambda k: datetime.strptime(k["setup"]["timestamp"], "%d/%m/%Y %H:%M:%S")
    )
    for plottype in ["mainLoop", "initializationTotal"]:
        data = (
            ["mainloop", "DynCore", "Remapping", "TracerAdvection"]
            if plottype == "mainLoop"
            else ["initialization", "total"]
        )
        plt.figure()
        for backend in ["python/gtx86", "python/numpy", "fortran", "python/gtcuda"]:
            specific = [x for x in alldata if x["setup"]["version"] == backend]
            if specific:
                for line in data:
                    plt.plot(
                        [
                            datetime.strptime(
                                e["setup"]["timestamp"], "%d/%m/%Y %H:%M:%S"
                            )
                            for e in specific
                        ],
                        [
                            e["times"][line]["mean"]
                            / (
                                (e["setup"]["timesteps"] - 1)
                                if plottype == "mainLoop"
                                else 1
                            )
                            for e in specific
                        ],
                        "--o",
                        label=line + " " + backend,
                    )
                    plt.fill_between(
                        [
                            datetime.strptime(
                                e["setup"]["timestamp"], "%d/%m/%Y %H:%M:%S"
                            )
                            for e in specific
                        ],
                        [
                            e["times"][line]["maximum"]
                            / (
                                (e["setup"]["timesteps"] - 1)
                                if plottype == "mainLoop"
                                else 1
                            )
                            for e in specific
                        ],
                        [
                            e["times"][line]["minimum"]
                            / (
                                (e["setup"]["timesteps"] - 1)
                                if plottype == "mainLoop"
                                else 1
                            )
                            for e in specific
                        ],
                        alpha=0.3,
                    )

        ax = plt.axes()
        ax.set_facecolor("silver")
        plt.gcf().autofmt_xdate()
        plt.ylabel(
            "Execution time per timestep"
            if plottype == "mainLoop"
            else "Execution time"
        )
        plt.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.05),
            ncol=2,
            fancybox=True,
            shadow=True,
            fontsize=8,
        )
        plt.title(plottype, pad=20)
        plt.figtext(
            0.5,
            0.01,
            "data: "
            + alldata[0]["setup"]["dataset"]
            + "  timesteps:"
            + str(alldata[0]["setup"]["timesteps"]),
            wrap=True,
            horizontalalignment="center",
            fontsize=12,
        )
        plt.grid(color="white", alpha=0.4)
        plt.savefig("history" + plottype + ".png")
