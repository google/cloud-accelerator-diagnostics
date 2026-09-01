# Copyright 2023 Google LLC
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#      https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runtime telemetry metric provider for TPU devices.

This module defines and registers metrics related to TPU hardware runtime,
memory usage, TensorCore duty cycles, and device idle durations with the central
MetricRegistry.
"""

from typing import Any

from tpu_info import cli_helper
from tpu_info.registry import register_metric
from rich import console


@register_metric(
    "hbm_usage",
    "runtime",
    "TPU HBM memory usage per device (GiB)",
)
def _hbm_usage_handler(
    *, chip_type: Any = None, count: int = 0, **_kwargs: Any
) -> list[console.RenderableType]:
  """Retrieves High Bandwidth Memory (HBM) usage tables for TPU devices.

  Queries current HBM memory consumption per device in GiB and formats the
  results into Rich renderable tables.

  Args:
    chip_type: The TPU chip type/generation (e.g., TpuChip.V4, TpuChip.V5e).
    count: The number of sample iterations to capture (0 for instantaneous).
    **_kwargs: Arbitrary keyword arguments forwarded by the registry.

  Returns:
    A list of Rich renderable tables displaying HBM memory usage.
  """
  return cli_helper.get_hbm_usage_table(chip_type, count)


@register_metric(
    "duty_cycle_percent",
    "runtime",
    "TPU TensorCore duty cycle percentage per core",
)
def _duty_cycle_handler(
    *, chip_type: Any = None, count: int = 0, **_kwargs: Any
) -> list[console.RenderableType]:
  """Retrieves TensorCore duty cycle percentage tables.

  Calculates the percentage of time TensorCores were actively executing
  workloads versus remaining idle across all available TPU cores.

  Args:
    chip_type: The TPU chip type/generation (e.g., TpuChip.V4, TpuChip.V5e).
    count: The number of sample iterations to capture (0 for instantaneous).
    **_kwargs: Arbitrary keyword arguments forwarded by the registry.

  Returns:
    A list of Rich renderable tables displaying TensorCore duty cycle
    percentages.
  """
  return cli_helper.get_duty_cycle_table(chip_type, count)


@register_metric(
    "tensorcore_utilization",
    "runtime",
    "TPU TensorCore utilization percentage from libtpu SDK",
)
def _tc_util_handler(
    *, count: int = 0, **_kwargs: Any
) -> list[console.RenderableType]:
  """Retrieves TensorCore utilization percentage tables from libtpu SDK.

  Interrogates the underlying libtpu runtime SDK to fetch low-level hardware
  TensorCore utilization metrics.

  Args:
    count: The number of sample iterations to capture.
    **_kwargs: Arbitrary keyword arguments forwarded by the registry.

  Returns:
    A list containing the rendered Rich table of TensorCore utilization.
  """
  return [cli_helper.TensorCoreUtilizationTable().render(count)]


@register_metric(
    "runtime_hbm_utilization",
    "runtime",
    "TPU HBM bandwidth utilization percentage per device",
)
def _runtime_hbm_util_handler(
    *, chip_type: Any = None, count: int = 0, **_kwargs: Any
) -> list[console.RenderableType]:
  """Retrieves runtime HBM memory bandwidth utilization tables.

  Measures memory bandwidth saturation and throughput percentage across TPU
  device HBM channels.

  Args:
    chip_type: The TPU chip type/generation (e.g., TpuChip.V4, TpuChip.V5e).
    count: The number of sample iterations to capture (0 for instantaneous).
    **_kwargs: Arbitrary keyword arguments forwarded by the registry.

  Returns:
    A list of Rich renderable tables displaying runtime HBM bandwidth
    utilization.
  """
  return cli_helper.get_runtime_hbm_utilization_table(chip_type, count)


@register_metric(
    "tensorcore_idle_duration",
    "runtime",
    "TPU TensorCore idle duration in seconds per device",
)
def _tc_idle_handler(
    *, chip_type: Any = None, count: int = 0, **_kwargs: Any
) -> list[console.RenderableType]:
  """Retrieves TensorCore idle duration tables.

  Tracks cumulative or instantaneous idle durations in seconds for TPU
  TensorCores across devices.

  Args:
    chip_type: The TPU chip type/generation (e.g., TpuChip.V4, TpuChip.V5e).
    count: The number of sample iterations to capture (0 for instantaneous).
    **_kwargs: Arbitrary keyword arguments forwarded by the registry.

  Returns:
    A list of Rich renderable tables displaying TensorCore idle durations.
  """
  return cli_helper.get_tensorcore_idle_duration_table(chip_type, count)
