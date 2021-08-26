import abc
from collections import OrderedDict
from typing import Any, Dict, List, Mapping, Set, Tuple, Union

import gt4py
from gt4py.definitions import BuildOptions
from gt4py.ir import ArgumentInfo, StencilDefinition, VarDecl
from gt4py.stencil_builder import StencilBuilder
from gt4py.stencil_object import StencilObject

from fv3core.utils.typing import Index3D


class Singleton(type):
    _instances: Dict[type, "Singleton"] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class StencilInterface(abc.ABC):
    def __init__(self):
        self.origin: Union[Index3D, Mapping[str, Tuple[int, ...]]] = None
        self.domain: Index3D = None
        self.build_info: Dict[str, Any] = {}

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def argument_names(self) -> Tuple[str, ...]:
        pass

    @abc.abstractmethod
    def set_argument_names(self, new_names: Tuple[str, ...]) -> None:
        pass

    @property
    @abc.abstractmethod
    def field_origins(self) -> Dict[str, Tuple[int, ...]]:
        pass

    @property
    @abc.abstractmethod
    def written_fields(self) -> List[str]:
        pass


def stencil_exists(stencil_group, stencil):
    for other_stencil in stencil_group:
        if stencil.name == other_stencil.name:
            return True
    return False


class StencilMerger(object, metaclass=Singleton):
    def __init__(self):
        self._stencil_groups: List[List[StencilInterface]] = []
        self._merged_groups: Dict[int, List[str]] = {}
        self._merged_stencils: Dict[int, StencilInterface] = {}
        self._saved_args: Dict[str, Dict[str, Any]] = {}

    def clear(self) -> None:
        self._stencil_groups.clear()

    def add(self, stencil: StencilInterface) -> None:
        assert "def_ir" in stencil.build_info
        if not self._stencil_groups:
            self._stencil_groups.append([])

        stencil_group = self._stencil_groups[-1]
        if stencil_group and (
            stencil.origin != stencil_group[-1].origin
            or stencil.domain != stencil_group[-1].domain
            or stencil_exists(stencil_group, stencil)
        ):
            stencil_group = []
            self._stencil_groups.append(stencil_group)

        stencil_group.append(stencil)

    def is_merged(self, stencil: StencilInterface) -> bool:
        for merged_names in self._merged_groups.values():
            if stencil.name in merged_names:
                return True
        return False

    def merged_position(self, stencil: StencilInterface) -> Tuple[int, int, int]:
        for group_id, merged_names in self._merged_groups.items():
            merged_index = merged_names.index(stencil.name)
            if merged_index >= 0:
                is_last = int(merged_index == len(merged_names) - 1)
                return (group_id, merged_index, is_last)
        return (-1, -1, 0)

    def merged_stencil(self, group_id: int) -> "StencilObject":
        return self._merged_stencils[group_id]

    def merge_args(self, group_id: int) -> Tuple[Tuple[Any, ...], Dict[Any, Any]]:
        all_arg_names = self._stencil_groups[group_id][0].argument_names
        merged_args: List[Any] = [None] * len(all_arg_names)
        merged_kwargs: Dict[Any, Any] = {}

        stencil_names = self._merged_groups[group_id]
        for stencil_name in stencil_names:
            saved_args = self._saved_args[stencil_name]
            merged_kwargs.update(saved_args["kwargs"])

            for arg_index, arg_name in enumerate(saved_args["arg_names"]):
                merged_index = all_arg_names.index(arg_name)
                merged_args[merged_index] = saved_args["args"][arg_index]

        return tuple(merged_args), merged_kwargs

    def save_args(self, stencil: StencilInterface, *args, **kwargs) -> None:
        self._saved_args[stencil.name].update({"args": args, "kwargs": kwargs})

    def merge(self) -> None:
        self._merged_groups.clear()
        for group_id, stencil_group in enumerate(self._stencil_groups):
            if len(stencil_group) > 1:
                for stencil in stencil_group:
                    self._saved_args[stencil.name] = dict(
                        arg_names=stencil.argument_names, args=[], kwargs={}
                    )

                top_stencil = stencil_group[0]
                top_ir = top_stencil.build_info["def_ir"]

                self._merged_groups[group_id] = [top_stencil.name]
                for next_stencil in stencil_group[1:]:
                    next_ir = next_stencil.build_info["def_ir"]
                    top_ir = self._merge_irs(top_ir, next_ir)

                    top_stencil.set_argument_names(
                        tuple([arg.name for arg in top_ir.api_signature])
                    )
                    top_stencil.field_origins.update(next_stencil.field_origins)
                    top_stencil.written_fields.extend(
                        [
                            written_field
                            for written_field in next_stencil.written_fields
                            if written_field not in top_stencil.written_fields
                        ]
                    )

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
        self, dest_ir: StencilDefinition, source_ir: StencilDefinition
    ) -> StencilDefinition:
        dest_ir.name = self._merge_names(dest_ir.name, source_ir.name)
        dest_ir.computations.extend(source_ir.computations)
        dest_ir.externals.update(source_ir.externals)
        dest_ir.api_fields = self._merge_named_lists(
            dest_ir.api_fields, source_ir.api_fields
        )
        dest_params: List[VarDecl] = self._merge_named_lists(
            dest_ir.parameters, source_ir.parameters
        )
        param_names = set([param.name for param in dest_params])
        dest_ir.parameters = dest_params
        dest_ir.api_signature = self._merge_api_signatures(
            dest_ir.api_signature, source_ir.api_signature, param_names
        )

        return dest_ir

    def _merge_names(self, dest_name: str, source_name: str) -> str:
        items = dest_name.split(".")
        module_name = ".".join(items[0:-1])
        dest_name = items[-1]
        source_name = source_name.split(".")[-1]
        return f"{module_name}.{dest_name}__{source_name}"

    def _merge_named_lists(
        self,
        dest_list: List[ArgumentInfo],
        source_list: List[ArgumentInfo],
    ) -> List[object]:
        dest_items = OrderedDict({item.name: item for item in dest_list})
        for item in source_list:
            if item.name not in dest_items:
                dest_items[item.name] = item

        return list(dest_items.values())

    def _merge_api_signatures(
        self,
        dest_list: List[ArgumentInfo],
        source_list: List[ArgumentInfo],
        param_names: Set[str],
    ) -> List[object]:
        dest_fields: Dict[str, ArgumentInfo] = OrderedDict()
        dest_params: Dict[str, ArgumentInfo] = OrderedDict()
        for item in dest_list:
            dest_dict = dest_params if item.name in param_names else dest_fields
            dest_dict[item.name] = item

        for item in source_list:
            dest_dict = dest_params if item.name in param_names else dest_fields
            if item.name not in dest_dict:
                dest_dict[item.name] = item

        new_api = dest_fields
        for param_name in dest_params:
            new_api[param_name] = dest_params[param_name]

        return list(new_api.values())
