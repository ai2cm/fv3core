# -*- coding: utf-8 -*-
import datetime as dt
import numpy as np

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Set, Union

import gt4py
from gt4py.definitions import BuildOptions, FieldInfo
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
        self._merged_groups = set()
        for group_id, stencil_group in enumerate(self._stencil_groups):
            if len(stencil_group) > 1:
                top_stencil = stencil_group[0]
                top_ir = top_stencil.build_info["def_ir"]

                for next_stencil in stencil_group[1:]:
                    next_ir = next_stencil.build_info["def_ir"]
                    top_ir = self._merge_irs(top_ir, next_ir)

                self._stencil_groups[group_id] = [top_stencil]
                self._merged_groups.add(group_id)

        self._rebuild()

    def _rebuild(self):
        for group_id, stencil_group in enumerate(self._stencil_groups):
            if group_id in self._merged_groups:
                top_stencil = stencil_group[0]
                stencil_object = top_stencil.stencil_object
                backend_class = gt4py.backend.from_name(stencil_object.backend)
                def_ir = top_stencil.build_info["def_ir"]

                stencil_options = stencil_object.options
                stencil_options["name"] = def_ir.name.split(".")[-1]
                stencil_options.pop("_impl_opts")

                builder = StencilBuilder(
                    top_stencil.definition_func,
                    backend=backend_class,
                    options=BuildOptions(**stencil_options),
                )
                builder.definition_ir = def_ir
                builder.externals = def_ir.externals

                stencil_class = builder.build()
                top_stencil.stencil_object = stencil_class()

    def _merge_irs(
        self, dest_ir: "StencilDefinition", source_ir: "StencilDefinition"
    ) -> "StencilDefinition":
        dest_ir.name = self._merge_names(dest_ir.name, source_ir.name)
        dest_ir.computations.extend(source_ir.computations)
        dest_ir.externals.update(source_ir.externals)
        dest_ir.api_fields = self._merge_named_lists(
            dest_ir.api_fields, source_ir.api_fields
        )
        dest_ir.parameters = self._merge_named_lists(
            dest_ir.parameters, source_ir.parameters
        )
        return dest_ir

    def _merge_names(self, dest_name: str, source_name: str) -> str:
        items = dest_name.split(".")
        module_name = ".".join(items[0:-1])
        dest_name = items[-1]
        source_name = source_name.split(".")[-1]
        return f"{module_name}.{dest_name}__{source_name}"

    def _merge_named_lists(
        self, dest_list: List[object], source_list: List[object]
    ) -> List[object]:
        dest_items = OrderedDict({item.name: item for item in dest_list})
        for item in source_list:
            if item.name not in dest_items:
                dest_items[item.name] = item
        return list(dest_items.values())
