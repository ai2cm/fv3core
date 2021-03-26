import statistics
from datetime import datetime
from argparse import ArgumentParser
import json
import os


def parse_args():
    usage = "usage: python %(prog)s <data_dir> <hash> "
    parser = ArgumentParser(usage=usage)
    parser.add_argument(
        "data_dir",
        type=str,
        action="store",
        help="directory containing run.daint.out file(s) from which data is collected",
    )
    parser.add_argument(
        "hash",
        type=str,
        action="store",
        help="git hash to store",
    )
    return parser.parse_args()


def gather_meta_data_from_line(output_line):
    """parses the output line that ran the dynamics to extract the dataset and the backend"""
    experiment_call = output_line.split("/")
    for index, path in enumerate(experiment_call):
        if "fv3core_serialized_test_data" in path:
            data_set = (experiment_call[index + 2]).split()[0]
            backend = "python/" + (experiment_call[index + 2]).split()[2]
            return data_set, backend


def gather_memory_usage_from_file(filename):
    """parses the output and collects data on gpu memory usage"""
    collected_data = {
        "data": [],
        "data_set": "",
        "backend": "",
    }
    with open(fullpath) as datafile:
        all_lines = datafile.readlines()
        for index, line in enumerate(all_lines):
            if "fv3core_serialized_test_data" in line:
                (
                    collected_data["data_set"],
                    collected_data["backend"],
                ) = gather_meta_data_from_line(line)
            elif "GPU utilization data" in line:
                # skip the three header lines
                data_lines = all_lines[index + 4 :]
                for line in data_lines:
                    splits = line.split()
                    if len(splits) > 1 and splits[1].isnumeric():
                        collected_data["data"].append(float(splits[1]))
                    else:
                        return collected_data
                return collected_data


def write_to_file(collected_data, git_hash):
    """writes statistics and metadata to a json file that is parsable by the plotting tool"""
    now = datetime.now()
    memory_footprint = {
        "minimum": min(collected_data["data"]),
        "maximum": max(collected_data["data"]),
        "mean": statistics.mean(collected_data["data"]),
        "setup": {
            "hash": git_hash,
            "timestamp": now.strftime("%d/%m/%Y %H:%M:%S"),
            "version": collected_data["backend"],
            "dataset": collected_data["data_set"],
        },
    }
    timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
    with open(timestamp + "_memory_usage.json", "w") as output:
        json.dump(memory_footprint, output, sort_keys=True, indent=4)


if __name__ == "__main__":
    args = parse_args()
    for subdir, dirs, files in os.walk(args.data_dir):
        for file in files:
            fullpath = os.path.join(subdir, file)
            if "run.daint.out" in fullpath:
                collected_data = gather_memory_usage_from_file(fullpath)
                write_to_file(collected_data, args.hash)
