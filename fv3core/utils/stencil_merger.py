# -*- coding: utf-8 -*-
import datetime as dt
import numpy as np

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Set, Union

from gt4py.definitions import FieldInfo
from gt4py.stencil_builder import StencilBuilder
from gt4py.stencil_object import StencilObject

from fv3core.decorators import FrozenStencil


class Container(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Container, cls).__call__(*args, **kwargs)
            cls._instances[cls].clear()
        return cls._instances[cls]


class StencilMerger(object, metaclass=Container):
    def __init__(self):
        self._stencil_groups: List[List[object]] = []

    def clear(self) -> None:
        self._stencil_groups.clear()

    def add(self, stencil: FrozenStencil) -> None:
        assert "def_ir" in stencil.build_info
        if not self._stencil_groups:
            self._stencil_groups.append([])

        stencil_group = self._stencil_groups[-1]
        if stencil_group and (
            stencil.origin != stencil_group[-1].origin
            or stencil.domain != stencil_group[-1].domain
        ):
            stencil_group = []
            self._stencil_groups.append(stencil_group)

        stencil_group.append(stencil)

    def merge(self) -> None:
        for group_id, stencil_group in enumerate(self._stencil_groups):
            top_stencil = stencil_group[0]
            top_ir = top_stencil.build_info["def_ir"]

            for next_stencil in stencil_group[1:]:
                next_ir = next_stencil.build_info["def_ir"]
                top_ir = self._merge_irs(top_ir, next_ir)

            top_stencil.build_info["def_ir"] = top_ir
            self._stencil_groups[group_id] = [top_stencil]

        # TODO(eddied): Merge stencil objects and regenerate code from merged IRs
        return

    def _merge_irs(self, dest_ir, source_ir) -> object:
        dest_ir.computations.extend(source_ir.computations)
        dest_ir.externals.update(source_ir.externals)
        dest_ir.api_fields = self._merge_named_lists(
            dest_ir.api_fields, source_ir.api_fields
        )
        dest_ir.parameters = self._merge_named_lists(
            dest_ir.parameters, source_ir.parameters
        )
        return dest_ir

    def _merge_named_lists(
        self, dest_list: List[object], source_list: List[object]
    ) -> List[object]:
        dest_items = OrderedDict({item.name: item for item in dest_list})
        for item in source_list:
            if item.name not in dest_items:
                dest_items[item.name] = item
        return list(dest_items.values())
