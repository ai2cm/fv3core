#!/usr/bin/env python
import csv
import io
import os
import pstats
# import sys
import warnings

from fv3gfs.util import CubedSpherePartitioner, TilePartitioner


def get_rank_location(partitioner, rank: int):
    tile_num = partitioner.tile_index(rank)
    left_edge = partitioner.tile.on_tile_left(rank)
    top_edge = partitioner.tile.on_tile_top(rank)
    right_edge = partitioner.tile.on_tile_right(rank)
    bot_edge = partitioner.tile.on_tile_bottom(rank)
    label = "C"
    if top_edge and left_edge:
        label = "NW"
    elif top_edge and right_edge:
        label = "NE"
    elif bot_edge and left_edge:
        label = "SW"
    elif bot_edge and right_edge:
        label = "SE"
    elif top_edge:
        label = "N"
    elif bot_edge:
        label = "S"
    elif left_edge:
        label = "W"
    elif right_edge:
        label = "E"
    return tile_num, label


def main():
    n_ranks = 96  # int(sys.argv[1]) if len(sys.argv) > 1 else 600
    backend = "gtc:cuda"
    backend_clean = backend.replace(":", "")
    experiment = "c384_96ranks_baroclinic"  # "test_data"
    data_path = "./data"

    partitioner = CubedSpherePartitioner(TilePartitioner(layout=(4, 4)))
    for rank in range(n_ranks):
        (tile, location) = get_rank_location(partitioner, rank)
        print(f"{rank},{tile},{location}")

    csv_file = f"{data_path}/fv3core_{n_ranks}ranks_{backend_clean}_profile.csv"
    with open(csv_file, "w+") as file:
        file.write("")

    for rank in range(n_ranks):
        # fv3core_test_data_gtc:cuda_336.prof
        file_path = f"{data_path}/fv3core_{experiment}_{backend}_{rank}.prof"
        if os.path.exists(file_path):
            out_stream = io.StringIO()
            stats = pstats.Stats(file_path, stream=out_stream)
            stats.strip_dirs()
            stats.print_stats()

            # chop off header lines
            result = "ncalls" + out_stream.getvalue().split("ncalls")[-1]
            lines = [
                line.rstrip().split(None, 5) for line in result.split("\n")
            ]

            lines[0].insert(0, "rank")
            lines[0].extend(
                "file_name,line_num,method_name,stencil_id,tile,location".split(",")
            )
            lines[0][3] += "_total"
            lines[0][6] = lines[0][6].replace(":", "_").replace("(", "_").replace(")", "")
            field_names = lines[0]
            n_cols = len(field_names)

            with open(csv_file, "a", newline='') as file:
                writer = csv.DictWriter(file, fieldnames=field_names)
                if rank == 0:
                    writer.writeheader()

                for i in range(1, len(lines)):
                    if lines[i]:
                        method_name = lines[i][-1].lstrip('{').rstrip('}')
                        file_name = "built-in"
                        line_num = 0
                        stencil_id = ""

                        if file_name in method_name:
                            method_name = method_name.split()[-1]
                            if "run_computation" in method_name:
                                stencil_id = method_name.split(".")[-2].split("_")[-2]
                        else:
                            if ":" in method_name:
                                file_name, line_and_method = method_name.split(":")
                                line_num, method_name = line_and_method.split("(")
                                method_name = method_name.rstrip(")")
                            if backend_clean in file_name:
                                stencil_id = file_name.split("_")[-1].replace(".py", "")

                        lines[i].insert(0, rank)
                        lines[i].extend([file_name, line_num, method_name, stencil_id])
                        (tile, location) = get_rank_location(partitioner, rank)
                        lines[i].extend([tile + 1, location])

                        assert(n_cols == len(lines[i]))
                        writer.writerow(dict(zip(field_names, lines[i])))
        else:
            warnings.warn(f"Missing data for R{rank}: '{file_path}' not found")

    print(f"Combined CSV file '{csv_file}' written")


if __name__ == "__main__":
    main()
