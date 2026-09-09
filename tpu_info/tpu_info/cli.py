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

import datetime
import sys
import time
from typing import Any

from tpu_info import args
from tpu_info import args_helper
from tpu_info import cli_helper
from tpu_info import device
from rich import box
from rich import console
from rich import live
from rich import panel
from rich import table as rich_table
from rich import text


# The minimum refresh rate in seconds, corresponding to a max of 30 FPS.
MIN_REFRESH_RATE_SECONDS = 1.0 / 30


def run_dashboard_streaming(
    rate: float,
    chip_type: Any,
    count: int,
    output_format: str,
    console_obj: console.Console,
):
  """Runs the summary dashboard in streaming mode."""
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

  if output_format != "table":
    print(
        f"Starting streaming mode (refresh rate: {effective_rate:.1f}s). Press"
        " Ctrl+C to exit.",
        file=sys.stderr,
    )
    try:
      while True:
        status_list, error_msg = cli_helper.fetch_consolidated_device_status(
            chip_type, count
        )
        if error_msg:
          print(f"WARNING: {error_msg}", file=sys.stderr)
        if output_format == "json":
          print(
              cli_helper.serialize_dashboard_to_json(
                  status_list, chip_type, count
              )
          )
        elif output_format == "csv":
          print(
              cli_helper.serialize_dashboard_to_csv(
                  status_list, chip_type
              ).strip()
          )
        time.sleep(effective_rate)
    except KeyboardInterrupt:
      print("\nExiting streaming mode.", file=sys.stderr)
    return

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
        console=console_obj,
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


def run_pmon_streaming(
    rate: float,
    chip_type: Any,
    count: int,
    output_format: str,
    console_obj: console.Console,
):
  """Runs the process monitor in streaming mode."""
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

  if output_format != "table":
    print(
        f"Starting streaming mode (refresh rate: {effective_rate:.1f}s). Press"
        " Ctrl+C to exit.",
        file=sys.stderr,
    )
    try:
      while True:
        status_list, _ = cli_helper.fetch_consolidated_device_status(
            chip_type, count
        )
        if output_format == "json":
          print(cli_helper.serialize_pmon_to_json(status_list, chip_type))
        elif output_format == "csv":
          print(
              cli_helper.serialize_pmon_to_csv(status_list, chip_type).strip()
          )
        time.sleep(effective_rate)
    except KeyboardInterrupt:
      print("\nExiting streaming mode.", file=sys.stderr)
    return

  print(
      f"Starting streaming mode (refresh rate: {effective_rate:.1f}s). Press"
      " Ctrl+C to exit."
  )
  data_refresh_hz = 1.0 / effective_rate
  target_screen_fps = data_refresh_hz * 1.2
  screen_refresh_per_second = min(max(4, int(target_screen_fps)), 30)

  try:
    status_list, _ = cli_helper.fetch_consolidated_device_status(
        chip_type, count
    )
    table = cli_helper.render_process_monitor_table(status_list, chip_type)
    with live.Live(
        table,
        console=console_obj,
        refresh_per_second=screen_refresh_per_second,
        screen=True,
        vertical_overflow="visible",
    ) as live_display:
      while True:
        time.sleep(effective_rate)
        status_list, _ = cli_helper.fetch_consolidated_device_status(
            chip_type, count
        )
        new_table = cli_helper.render_process_monitor_table(
            status_list, chip_type
        )
        live_display.update(new_table)
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

  output_format = getattr(cli_args, "format", "table")
  if is_incompatible and output_format == "table":
    console_obj.print(cli_helper.get_py_compat_warning_panel())

  # Handle global version check
  if cli_args.version:
    if output_format == "table":
      print(f"- tpu-info version: {cli_helper.fetch_cli_version()}")
      if is_incompatible:
        print("- libtpu version: N/A (incompatible environment)")
        print("- accelerator type: N/A (incompatible environment)")
      else:
        print(f"- libtpu version: {cli_helper.fetch_libtpu_version()}")
        print(f"- accelerator type: {cli_helper.fetch_accelerator_type()}")
    elif output_format == "json":
      import json as std_json  # pylint: disable=g-import-not-at-top
      print(
          std_json.dumps(
              {
                  "tpu_info_version": cli_helper.fetch_cli_version(),
                  "libtpu_version": (
                      "N/A"
                      if is_incompatible
                      else cli_helper.fetch_libtpu_version()
                  ),
                  "accelerator_type": (
                      "N/A"
                      if is_incompatible
                      else cli_helper.fetch_accelerator_type()
                  ),
              },
              indent=2,
          )
      )
    elif output_format == "csv":
      import csv as std_csv  # pylint: disable=g-import-not-at-top
      import io as std_io  # pylint: disable=g-import-not-at-top
      out = std_io.StringIO()
      w = std_csv.writer(out)
      w.writerow(["tpu_info_version", "libtpu_version", "accelerator_type"])
      w.writerow([
          cli_helper.fetch_cli_version(),
          "N/A" if is_incompatible else cli_helper.fetch_libtpu_version(),
          "N/A" if is_incompatible else cli_helper.fetch_accelerator_type(),
      ])
      print(out.getvalue().strip())
    return

  subcommand = cli_args.subcommand

  # Handle list metrics before checking local chips
  if subcommand == "query" and cli_args.list_metrics:
    if output_format == "table":
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
    elif output_format == "json":
      print(cli_helper.serialize_metrics_list_to_json())
    elif output_format == "csv":
      print(cli_helper.serialize_metrics_list_to_csv())
    return

  chip_type, count = device.get_local_chips()
  if not chip_type:
    print("No TPU chips found.", file=sys.stderr)
    return

  if subcommand == "dashboard":
    if cli_args.streaming:
      run_dashboard_streaming(
          cli_args.rate, chip_type, count, output_format, console_obj
      )
    else:
      if output_format == "table":
        dashboard_group = cli_helper.get_dashboard(chip_type, count)
        console_obj.print(dashboard_group)
      elif output_format == "json":
        status_list, error_msg = cli_helper.fetch_consolidated_device_status(
            chip_type, count
        )
        if error_msg:
          print(f"WARNING: {error_msg}", file=sys.stderr)
        print(
            cli_helper.serialize_dashboard_to_json(
                status_list, chip_type, count
            )
        )
      elif output_format == "csv":
        status_list, error_msg = cli_helper.fetch_consolidated_device_status(
            chip_type, count
        )
        if error_msg:
          print(f"WARNING: {error_msg}", file=sys.stderr)
        print(cli_helper.serialize_dashboard_to_csv(status_list, chip_type))

  elif subcommand == "pmon":
    if cli_args.streaming:
      run_pmon_streaming(
          cli_args.rate, chip_type, count, output_format, console_obj
      )
    else:
      status_list, _ = cli_helper.fetch_consolidated_device_status(
          chip_type, count
      )
      if output_format == "table":
        table = cli_helper.render_process_monitor_table(status_list, chip_type)
        console_obj.print(table)
      elif output_format == "json":
        print(cli_helper.serialize_pmon_to_json(status_list, chip_type))
      elif output_format == "csv":
        print(cli_helper.serialize_pmon_to_csv(status_list, chip_type))

  elif subcommand == "query":
    group_arg = getattr(cli_args, "group", None)
    if not (cli_args.metric or group_arg):
      print(
          "ERROR: No metrics or groups specified. Use '--metric <name>',"
          " '-g/--group <group>', or '--list-metrics'.",
          file=sys.stderr,
      )
      sys.exit(1)

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

      if output_format == "table":
        renderables = cli_helper.fetch_metric_tables(
            validated_metrics, chip_type, count
        )
        for item in renderables:
          console_obj.print(item)
      elif output_format == "json":
        import json as std_json  # pylint: disable=g-import-not-at-top
        from tpu_info.registry import MetricRegistry  # pylint: disable=g-import-not-at-top

        utc_time = datetime.datetime.now(datetime.timezone.utc)
        timestamp_str = utc_time.isoformat()

        metrics_data = {}
        registry = MetricRegistry()
        for metric_req in validated_metrics:
          metric_name = metric_req[0]
          filters = metric_req[1]
          if filters:
            formatted_parts = []
            for k, v in sorted(filters.items()):
              if isinstance(v, list):
                formatted_parts.append(f"{k}=[{','.join(str(x) for x in v)}]")
              else:
                formatted_parts.append(f"{k}={v}")
            filter_str = ",".join(formatted_parts)
            metric_key = f"{metric_name}[{filter_str}]"
          else:
            metric_key = metric_name
          try:
            raw_data = registry.get_raw_metric(
                metric_name, chip_type, count, filters
            )
            metrics_data[metric_key] = raw_data
          except Exception as e:  # pylint: disable=broad-exception-caught
            # Log telemetry extraction error to stderr
            print(
                f"WARNING: Failed to fetch raw metric '{metric_key}': {e}",
                file=sys.stderr,
            )
            metrics_data[metric_key] = f"ERROR: {e}"

        print(
            std_json.dumps(
                {"timestamp": timestamp_str, "metrics": metrics_data}, indent=2
            )
        )
      elif output_format == "csv":
        renderables = cli_helper.fetch_metric_tables(
            validated_metrics, chip_type, count
        )
        tables = []
        for item in renderables:
          if isinstance(item, rich_table.Table):
            tables.append(item)
          elif isinstance(item, panel.Panel):
            # Strip rich tags and print warning to stderr
            import re as std_re  # pylint: disable=g-import-not-at-top
            renderable_content = getattr(item, "renderable", item)
            if hasattr(renderable_content, "plain"):
              plain_text = renderable_content.plain
            else:
              try:
                plain_text = text.Text.from_markup(
                    str(renderable_content)
                ).plain
              except Exception:  # pylint: disable=broad-exception-caught
                plain_text = std_re.sub(
                    r"\[\/?[^\]]+\]", "", str(renderable_content)
                )
            plain_text = std_re.sub(r"\[\/?[^\]]+\]", "", plain_text)
            print(f"WARNING: {plain_text}", file=sys.stderr)
        print(cli_helper.serialize_tables_to_csv(tables), end="")

    except args_helper.MetricParsingError as e:
      if output_format == "table":
        console_obj.print(
            panel.Panel(
                text.Text(str(e), style="red"),
                title="[b]Metric Parsing Error[/b]",
                border_style="red",
                box=box.ASCII,
            )
        )
      else:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

  elif subcommand == "topo":
    if output_format == "table":
      table = cli_helper.fetch_topo_table()
      console_obj.print(table)
    elif output_format == "json":
      print(cli_helper.serialize_topo_to_json())
    elif output_format == "csv":
      print(cli_helper.serialize_topo_to_csv())
