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

"""Command line interface for tpu-info."""

import sys
import time
from typing import Any

from tpu_info import args
from tpu_info import args_helper
from tpu_info import cli_helper
from tpu_info import device
from rich import console
from rich import live
from rich import panel
from rich import text


# The minimum refresh rate in seconds, corresponding to a max of 30 FPS.
MIN_REFRESH_RATE_SECONDS = 1.0 / 30


def run_dashboard_streaming(
    rate: float,
    chip_type: Any,
    count: int,
    console_obj: console.Console,
):
  """Runs the summary dashboard in streaming mode."""
  del console_obj  # Unused by streaming Live display.
  if rate <= 0:
    print("Error: Refresh rate must be positive.", file=sys.stderr)
    return

  effective_rate = rate
  if rate < MIN_REFRESH_RATE_SECONDS:
    console_err = console.Console(stderr=True)
    console_err.print(
        f"[yellow]WARNING: Provided rate {rate:.3f}s is faster than"
        " the supported maximum. Capping at"
        f" {MIN_REFRESH_RATE_SECONDS:.3f}s.[/yellow]"
    )
    effective_rate = MIN_REFRESH_RATE_SECONDS

  print(
      f"Starting streaming mode (refresh rate: {effective_rate:.1f}s). Press"
      " Ctrl+C to exit."
  )
  data_refresh_hz = 1.0 / effective_rate
  target_screen_fps = data_refresh_hz * 1.2
  screen_refresh_per_second = min(max(4, int(target_screen_fps)), 30)

  try:
    dashboard = cli_helper.get_dashboard(chip_type, count, effective_rate)

    with live.Live(
        dashboard,
        refresh_per_second=screen_refresh_per_second,
        screen=True,
        vertical_overflow="visible",
    ) as live_display:
      while True:
        time.sleep(effective_rate)
        new_dashboard = cli_helper.get_dashboard(
            chip_type, count, effective_rate
        )
        live_display.update(new_dashboard)
  except KeyboardInterrupt:
    print("\nExiting streaming mode.")
  except Exception as e:
    print(
        "\nFATAL ERROR during streaming update cycle, stopping stream:"
        f" {type(e).__name__}: {e}",
        file=sys.stderr,
    )
    raise e


def print_chip_info():
  """Print local TPU devices and libtpu runtime metrics."""
  try:
    cli_args = args.parse_arguments()
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(f"Error parsing arguments: {e}", file=sys.stderr)
    sys.exit(1)

  console_obj = console.Console()
  is_incompatible = cli_helper.is_incompatible_python_version()

  if is_incompatible:
    console_obj.print(cli_helper.get_py_compat_warning_panel())

  # Handle global version check
  if cli_args.version:
    print(f"- tpu-info version: {cli_helper.fetch_cli_version()}")
    if is_incompatible:
      print("- libtpu version: N/A (incompatible environment)")
      print("- accelerator type: N/A (incompatible environment)")
    else:
      print(f"- libtpu version: {cli_helper.fetch_libtpu_version()}")
      print(f"- accelerator type: {cli_helper.fetch_accelerator_type()}")
    return

  # Handle list metrics before checking local chips
  if cli_args.list_metrics:
    from tpu_info.registry import MetricRegistry  # pylint: disable=g-import-not-at-top
    from rich.tree import Tree  # pylint: disable=g-import-not-at-top

    registry = MetricRegistry()
    root_tree = Tree(
        "[bold]Supported Telemetry Metrics[/bold]", guide_style="cyan"
    )
    for group in registry.get_all_groups():
      descriptors = registry.get_descriptors_by_group(group)
      group_node = root_tree.add(
          f"[bold green]{group}[/bold green] ([yellow]{len(descriptors)}"
          " metrics[/yellow])"
      )
      for desc in sorted(descriptors, key=lambda d: d.name):
        group_node.add(f"[cyan]{desc.name}[/cyan]: {desc.description}")
    console_obj.print(root_tree)
    return

  chip_type, count = device.get_local_chips()
  if not chip_type:
    print("No TPU chips found.", file=sys.stderr)
    return

  # Handle flat process list
  if cli_args.process:
    table = cli_helper.fetch_process_table(chip_type, count)
    console_obj.print(table)
    return

  # Handle flat metric/group query
  group_arg = getattr(cli_args, "group", None)
  if cli_args.metric or group_arg:
    try:
      metric_requests = list(cli_args.metric) if cli_args.metric else []
      if group_arg:
        from tpu_info.registry import MetricRegistry  # pylint: disable=g-import-not-at-top

        registry = MetricRegistry()
        for g in group_arg:
          if g not in registry.get_all_groups():
            raise args_helper.MetricParsingError(
                f"ERROR: Invalid group '{g}'. Supported groups:"
                f" {', '.join(registry.get_all_groups())}"
            )
          for desc in sorted(
              registry.get_descriptors_by_group(g), key=lambda d: d.name
          ):
            if not any(m.name == desc.name for m in metric_requests):
              metric_requests.append(args.MetricRequest(name=desc.name))
      validated_metrics = args_helper.MetricsParser.parse_metric_args(
          metric_requests
      )

      renderables = cli_helper.fetch_metric_tables(
          validated_metrics, chip_type, count
      )
      for item in renderables:
        console_obj.print(item)

    except args_helper.MetricParsingError as e:
      console_obj.print(
          panel.Panel(
              text.Text(str(e), style="red"),
              title="[b]Metric Parsing Error[/b]",
              border_style="red",
          )
      )
      sys.exit(1)
    return

  # Default is dashboard
  if cli_args.streaming:
    run_dashboard_streaming(cli_args.rate, chip_type, count, console_obj)
  else:
    dashboard_group = cli_helper.get_dashboard(chip_type, count)
    console_obj.print(dashboard_group)
