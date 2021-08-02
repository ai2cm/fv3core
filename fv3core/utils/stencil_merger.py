# -*- coding: utf-8 -*-
from collections import OrderedDict
from typing import Any, Dict, List, Tuple

import gt4py
from gt4py.definitions import BuildOptions
from gt4py.stencil_builder import StencilBuilder
from gt4py.stencil_object import StencilObject

# from fv3core.decorators import FrozenStencil


class Container(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Container, cls).__call__(*args, **kwargs)
            cls._instances[cls].clear()
        return cls._instances[cls]


class StencilMerger(object, metaclass=Container):
    def __init__(self):
        self._stencil_groups: List[List["FrozenStencil"]] = []
        self._merged_groups: Dict[int, List[str]] = {}
        self._merged_stencils: Dict[int, "FrozenStencil"] = {}
        self._saved_args: Dict[str, Dict[str, Any]] = {}

    def clear(self) -> None:
        self._stencil_groups.clear()

    def add(self, stencil: "FrozenStencil") -> None:
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

    def is_merged(self, stencil: "FrozenStencil") -> bool:
        for merged_names in self._merged_groups.values():
            if stencil.name in merged_names:
                return True
        return False

    def merged_position(self, stencil: "FrozenStencil") -> Tuple[int, int, int]:
        for group_id, merged_names in self._merged_groups.items():
            merged_index = merged_names.index(stencil.name)
            if merged_index >= 0:
                is_last = int(merged_index == len(merged_names) - 1)
                return (group_id, merged_index, is_last)
        return (-1, -1, 0)

    def merged_stencil(self, group_id: int) -> "StencilObject":
        return self._merged_stencils[group_id]

    def merge_args(self, group_id: int) -> Tuple[List[Any], Dict[str, Any]]:
        stencil_names = self._merged_groups[group_id]
        merged_args = list(self._saved_args[stencil_names[0]]["args"])
        merged_kwargs = self._saved_args[stencil_names[0]]["kwargs"]

        for i in range(1, len(stencil_names)):
            args = self._saved_args[stencil_names[i]]["args"]
            for arg in args:
                arg_found: bool = False
                for merged_arg in merged_args:
                    arg_found = id(arg) == id(merged_arg)
                    if arg_found:
                        break
                if not arg_found:
                    merged_args.append(arg)

            kwargs = self._saved_args[stencil_names[i]]["kwargs"]
            merged_kwargs.update({name: value for name, value in kwargs.items()})

        self._saved_args.clear()
        return merged_args, merged_kwargs

    def save_args(self, stencil: "FrozenStencil", *args, **kwargs) -> None:
        self._saved_args[stencil.name] = dict(args=args, kwargs=kwargs)

    def merge(self) -> None:
        self._merged_groups.clear()
        for group_id, stencil_group in enumerate(self._stencil_groups):
            if len(stencil_group) > 1:
                top_stencil = stencil_group[0]
                top_ir = top_stencil.build_info["def_ir"]

                self._merged_groups[group_id] = [top_stencil.name]
                for next_stencil in stencil_group[1:]:
                    next_ir = next_stencil.build_info["def_ir"]
                    top_ir = self._merge_irs(top_ir, next_ir)
                    arg_names = [
                        arg_name
                        for arg_name in next_stencil._argument_names
                        if arg_name not in top_stencil._argument_names
                    ]

                    top_stencil._argument_names += tuple(arg_names)
                    top_stencil._field_origins.update(next_stencil._field_origins)
                    written_fields = [
                        written_field
                        for written_field in next_stencil._written_fields
                        if written_field not in top_stencil._written_fields
                    ]
                    top_stencil._written_fields.extend(written_fields)

                    next_stencil._argument_names = top_stencil._argument_names
                    next_stencil._field_origins = top_stencil._field_origins
                    next_stencil._written_fields = top_stencil._written_fields

                    self._merged_groups[group_id].append(next_stencil.name)

                self._stencil_groups[group_id] = [top_stencil]

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
                self._merged_stencils[group_id] = stencil_class()

    def _merge_irs(
        self, dest_ir: "StencilDefinition", source_ir: "StencilDefinition"
    ) -> "StencilDefinition":
        dest_ir.name = self._merge_names(dest_ir.name, source_ir.name)
        dest_ir.computations.extend(source_ir.computations)
        dest_ir.externals.update(source_ir.externals)
        dest_ir.api_signature = self._merge_named_lists(
            dest_ir.api_signature, source_ir.api_signature
        )
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
