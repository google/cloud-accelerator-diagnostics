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
from tpu_info import device
from tpu_info import metrics
from tpu_info.registry import register_metric
from rich import console


def _hbm_usage_raw(chip_type: Any = None, **_kwargs: Any) -> Any:
  """Retrieves raw HBM memory usage data."""
  return (
      metrics.get_chip_usage_new(chip_type)
      if chip_type is device.TpuChip.V7X
      else metrics.get_chip_usage(chip_type)
  )


def _duty_cycle_raw(chip_type: Any = None, **_kwargs: Any) -> Any:
  """Retrieves raw duty cycle data across TPU cores."""
  usages = (
      metrics.get_chip_usage_new(chip_type)
      if chip_type is device.TpuChip.V7X
      else metrics.get_chip_usage(chip_type)
  )
  return [
      {"device_id": u.device_id, "duty_cycle_pct": u.duty_cycle_pct}
      for u in usages
  ]


def _tc_util_raw(count: int = 0, **_kwargs: Any) -> Any:
  """Retrieves raw TensorCore utilization data from libtpu SDK."""
  del count  # Unused.
  cli_helper.ensure_libtpu_initialized()
  if cli_helper.libtpu_sdk is None:
    return []
  sdk = cli_helper.libtpu_sdk
  monitoring_module = getattr(sdk, "tpumonitoring", None) or getattr(
      sdk, "monitoring", None
  )
  if monitoring_module:
    data = monitoring_module.get_metric("tensorcore_util").data()
    return [{"core_id": i, "utilization": float(v)} for i, v in enumerate(data)]
  return []


def _runtime_hbm_util_raw(**_kwargs: Any) -> Any:
  """Retrieves raw runtime HBM bandwidth utilization data."""
  return [
      {"device_id": d, "utilization": v}
      for d, v in metrics.get_runtime_hbm_utilization()
  ]


def _tc_idle_raw(**_kwargs: Any) -> Any:
  """Retrieves raw TensorCore idle duration data."""
  return [
      {"device_id": d, "idle_duration_seconds": v}
      for d, v in metrics.get_tensorcore_idle_duration()
  ]


@register_metric(
    "hbm_usage",
    "runtime",
    "TPU HBM memory usage per device (GiB)",
    raw_handler=_hbm_usage_raw,
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
    raw_handler=_duty_cycle_raw,
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
    raw_handler=_tc_util_raw,
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
    raw_handler=_runtime_hbm_util_raw,
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
    raw_handler=_tc_idle_raw,
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
