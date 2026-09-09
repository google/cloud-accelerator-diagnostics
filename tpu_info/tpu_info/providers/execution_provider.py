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

"""Execution telemetry metric provider for TPU devices.

This module defines and registers metrics related to XLA HLO compilation,
execution queues, timing distributions, and low-level TPUz sequencer and core
states with the central MetricRegistry.
"""

from typing import Any

from tpu_info import cli_helper
from tpu_info import metrics
from tpu_info.registry import register_metric
from rich import console


def _hlo_queue_size_raw(chip_type: Any = None, **_kwargs: Any) -> Any:
  """Retrieves raw HLO queue size data per TPU device."""
  return metrics.get_hlo_queue_size(chip_type)


def _hlo_exec_timing_raw(chip_type: Any = None, **_kwargs: Any) -> Any:
  """Retrieves raw HLO execution timing data per TPU device."""
  return metrics.get_hlo_exec_timing(chip_type)


def _core_state_raw(**_kwargs: Any) -> Any:
  """Retrieves raw TPUz core states data."""
  core_states = metrics.get_tpuz_info(include_hlo_info=False)
  return [
      {
          "chip_id": c.chip_id,
          "global_core_id": c.global_core_id,
          "core_type": c.core_type,
          "xdb_server": c.xdb_server,
      }
      for c in core_states
  ]


def _sequencer_state_raw(**_kwargs: Any) -> Any:
  """Retrieves raw TPUz sequencer states summary data."""
  core_states = metrics.get_tpuz_info(include_hlo_info=False)
  res = []
  for c in core_states:
    for s in c.sequencer_states:
      res.append({
          "chip_id": c.chip_id,
          "global_core_id": c.global_core_id,
          "program_counter": s.pc,
          "tracemark": s.tracemark,
          "program_id": s.program_id,
          "run_id": s.run_id,
          "sequence_type": s.sequencer_type,
      })
  return res


def _sequencer_state_detailed_raw(**_kwargs: Any) -> Any:
  """Retrieves raw detailed TPUz sequencer states data."""
  core_states = metrics.get_tpuz_info(include_hlo_info=True)
  res = []
  for c in core_states:
    for s in c.sequencer_states:
      res.append({
          "chip_id": c.chip_id,
          "global_core_id": c.global_core_id,
          "program_counter": s.pc,
          "tracemark": s.tracemark,
          "program_id": s.program_id,
          "run_id": s.run_id,
          "sequence_type": s.sequencer_type,
          "core_error": c.error_message,
          "hlo_location": s.hlo_location,
          "hlo_details": s.hlo_detailed_info,
      })
  return res


def _queued_programs_raw(**_kwargs: Any) -> Any:
  """Retrieves raw TPUz queued programs data."""
  core_states = metrics.get_tpuz_info(include_hlo_info=False)
  res = []
  for c in core_states:
    for p in c.queued_programs:
      res.append({
          "chip_id": c.chip_id,
          "global_core_id": c.global_core_id,
          "run_id": p.run_id,
          "launch_id": p.launch_id,
          "program_fingerprint": p.program_fingerprint,
      })
  return res


@register_metric(
    "hlo_queue_size",
    "execution",
    "HLO queue size gauge per device",
    raw_handler=_hlo_queue_size_raw,
)
def _hlo_queue_size_handler(
    *, chip_type: Any = None, count: int = 0, **_kwargs: Any
) -> list[console.RenderableType]:
  """Retrieves High-Level Optimizer (HLO) execution queue size tables.

  Monitors the depth of pending XLA HLO module execution queues per TPU device.

  Args:
    chip_type: The TPU chip type/generation (e.g., TpuChip.V4, TpuChip.V5e).
    count: The number of sample iterations to capture.
    **_kwargs: Arbitrary keyword arguments forwarded by the registry.

  Returns:
    A list of Rich renderable tables displaying HLO queue sizes.
  """
  return cli_helper.get_hlo_queue_size_table(chip_type, count)


@register_metric(
    "hlo_exec_timing",
    "execution",
    "HLO execution timing distribution in microseconds",
    raw_handler=_hlo_exec_timing_raw,
)
def _hlo_exec_timing_handler(
    *, chip_type: Any = None, count: int = 0, **_kwargs: Any
) -> list[console.RenderableType]:
  """Retrieves HLO execution timing distribution tables.

  Collects latency distributions in microseconds for HLO program execution on
  TPU chips.

  Args:
    chip_type: The TPU chip type/generation (e.g., TpuChip.V4, TpuChip.V5e).
    count: The number of sample iterations to capture.
    **_kwargs: Arbitrary keyword arguments forwarded by the registry.

  Returns:
    A list of Rich renderable tables displaying HLO execution timing
    distributions.
  """
  return cli_helper.get_hlo_exec_timing_table(chip_type, count)


@register_metric(
    "core_state",
    "execution",
    "TPUz core states (Chip ID, Global Core ID, Core Type, xdb Server)",
    raw_handler=_core_state_raw,
)
def _core_state_handler(**_kwargs: Any) -> list[console.RenderableType]:
  """Retrieves TPUz hardware core states table.

  Interrogates the TPUz debug server for low-level core operational states,
  including Chip ID, Global Core ID, Core Type, and xdb Server location.

  Args:
    **_kwargs: Arbitrary keyword arguments forwarded by the registry.

  Returns:
    A list of Rich renderable tables displaying TPUz core states.
  """
  return cli_helper.get_tpuz_core_state()


@register_metric(
    "sequencer_state",
    "execution",
    "TPUz sequencer states (PC, Tracemark, Program ID, Run ID, Sequence Type)",
    raw_handler=_sequencer_state_raw,
)
def _sequencer_state_handler(**_kwargs: Any) -> list[console.RenderableType]:
  """Retrieves TPUz sequencer states summary table.

  Collects high-level execution sequencer status including Program Counter (PC),
  tracemark, Program ID, Run ID, and Sequence Type.

  Args:
    **_kwargs: Arbitrary keyword arguments forwarded by the registry.

  Returns:
    A list of Rich renderable tables displaying summary sequencer states.
  """
  return cli_helper.get_tpuz_sequencer_state(detailed_info=False)


@register_metric(
    "sequencer_state_detailed",
    "execution",
    "Detailed TPUz sequencer states including HLO location and errors",
    raw_handler=_sequencer_state_detailed_raw,
)
def _sequencer_state_detailed_handler(
    **_kwargs: Any,
) -> list[console.RenderableType]:
  """Retrieves detailed TPUz sequencer states table.

  Collects in-depth execution sequencer diagnostics including source HLO
  instructions, call stack locations, and trapped hardware error states.

  Args:
    **_kwargs: Arbitrary keyword arguments forwarded by the registry.

  Returns:
    A list of Rich renderable tables displaying detailed sequencer states.
  """
  return cli_helper.get_tpuz_sequencer_state(detailed_info=True)


@register_metric(
    "queued_programs",
    "execution",
    "TPUz queued programs (Run ID, Launch ID, Program Fingerprint)",
    raw_handler=_queued_programs_raw,
)
def _queued_programs_handler(**_kwargs: Any) -> list[console.RenderableType]:
  """Retrieves TPUz queued programs table.

  Lists queued and currently executing TPU programs along with Run IDs,
  Launch IDs, and Program Fingerprints.

  Args:
    **_kwargs: Arbitrary keyword arguments forwarded by the registry.

  Returns:
    A list of Rich renderable tables displaying queued program information.
  """
  return cli_helper.get_tpuz_queued_programs()

