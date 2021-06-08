from typing import Any, Collection, Dict
import re

REPORT_KEY_COUNT = -1
REPORT_DURATION_INDEX = 1
REPORT_GRDX_INDEX = 2
REPORT_BLKX_INDEX = 3
REPORT_NAME_INDEX = 4


def _guard_gpu_report_row(config: Dict, gpu_report_row: Collection[Any]):
    if len(gpu_report_row) != config[REPORT_KEY_COUNT]:
        raise RuntimeError("Bad row size, is it a gputrace report row?")


def _is_kernel(config: Dict, gpu_report_row) -> bool:
    # Test for kernel by looking at grid/block size being there
    if (
        gpu_report_row[config[REPORT_GRDX_INDEX]] is None
        and gpu_report_row[config[REPORT_BLKX_INDEX]] is None
    ):
        return False

    return True


def filter_kernel_name(
    config: Dict, gpu_report_row: Collection[Any]
) -> Collection[Any]:
    if gpu_report_row is None:
        return None
    _guard_gpu_report_row(config, gpu_report_row)

    if not _is_kernel(config, gpu_report_row):
        return gpu_report_row

    # Run a query to convert the stencil generated string to a readable one
    approx_stencil_name_re = re.search(
        "(?<=bound_functorIN)(.*)(?=___gtcuda)",
        gpu_report_row[config[REPORT_NAME_INDEX]],
    )
    if approx_stencil_name_re is None:
        return gpu_report_row
    approx_stencil_name = approx_stencil_name_re.group().lstrip("0123456789 ")
    row_as_list = list(gpu_report_row)
    row_as_list[config[REPORT_NAME_INDEX]] = approx_stencil_name
    return tuple(row_as_list)


def filter_kernel_only(
    config: Dict, gpu_report_row: Collection[Any]
) -> Collection[Any]:
    _guard_gpu_report_row(config, gpu_report_row)

    if not _is_kernel(gpu_report_row):
        return None

    return gpu_report_row


def filter_kernel_time_under_threshold(
    config: Dict, gpu_report_row: Collection[Any], threshold_in_ns: int
) -> Collection[Any]:
    if gpu_report_row is None:
        return None

    _guard_gpu_report_row(config, gpu_report_row)

    if not _is_kernel(gpu_report_row):
        return None

    if gpu_report_row[config[REPORT_DURATION_INDEX]] < threshold_in_ns:
        return gpu_report_row
    else:
        return None
