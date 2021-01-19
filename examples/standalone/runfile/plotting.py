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
        help="directory containing data to run with",
    )
    args = parser.parse_args()

    # collect and sort the data
    alldata = []
    for filename in os.listdir(args.data_dir):
        if filename.endswith(".json"):
            print(filename)
            fullpath = args.data_dir + "/" + filename
            with open(fullpath) as f:
                alldata.append(json.load(f))
    alldata.sort(
        key=lambda k: datetime.strptime(
            k["setup"]["experiment time"], "%d/%m/%Y %H:%M:%S"
        )
    )

    # simple plots for the history
    for data in ["main", "init", "cleanup", "total"]:
        if "mean" in alldata[0]["times"][data]:
            plt.plot(
                [e["setup"]["experiment time"] for e in alldata],
                [e["times"][data]["mean"] for e in alldata],
                label="main loop:" + data,
            )
            plt.fill_between(
                [e["setup"]["experiment time"] for e in alldata],
                [e["times"][data]["maximum"] for e in alldata],
                [e["times"][data]["minimum"] for e in alldata],
                alpha=0.5,
            )

    plt.gcf().autofmt_xdate()
    plt.ylabel("Execution time")
    plt.legend()
    plt.savefig("history.png")
