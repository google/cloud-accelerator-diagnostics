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

"""Argument parsing for the tpu-info tool."""

import argparse
import dataclasses
import sys


@dataclasses.dataclass
class MetricRequest:
  """Represents a user's request for a single metric from the CLI.

  This class is used by the argument parser to create a link
  between a `--metric` flag and its associated filter string
  from a `-f` flag. A list of these objects is the final output of
  the `parse_arguments()` function, which is then passed on for validation
  and processing.
  """

  name: str
  filter_str: str | None = None


class _MetricAndFilterAction(argparse.Action):
  """Custom action to associate --metric and --filter flags."""

  def __call__(self, parser, namespace, values, option_string=None):
    # Initialize the list on first use.
    if (
        not hasattr(namespace, self.dest)
        or getattr(namespace, self.dest) is None
    ):
      setattr(namespace, self.dest, [])

    metrics_list = getattr(namespace, self.dest)

    if option_string == "--metric":
      # Support space-separated and comma-separated lists of metrics
      for val in values:
        for name in val.split(","):
          name = name.strip()
          if name:
            metrics_list.append(MetricRequest(name=name))
    elif option_string in ("-f", "--filter"):
      if not metrics_list:
        raise argparse.ArgumentError(
            self, f"{option_string} must be used after --metric."
        )
      last_metric = metrics_list[-1]
      if last_metric.filter_str is not None:
        raise argparse.ArgumentError(
            self, f"Only one filter is allowed per metric ({last_metric.name})."
        )
      last_metric.filter_str = values


def preprocess_legacy_args(argv: list[str]) -> list[str]:
  """Pre-parser detection for legacy v0.x flags.

  Routes legacy flags to appropriate subparsers and prints a deprecation warning
  to stderr.

  Args:
    argv: List of command line arguments (excluding script name).

  Returns:
    Rewritten list of command line arguments.
  """
  if not argv:
    return ["dashboard"]

  subcommands = {"dashboard", "pmon", "query", "topo"}

  # If a subcommand is already present, ensure it is the first positional
  # argument so subparser options (and preceding global options) are routed
  # correctly without false deprecation warnings.
  for i, arg in enumerate(argv):
    if arg in subcommands:
      if i == 0:
        return argv
      return [arg] + argv[:i] + argv[i + 1 :]

  # If global help or version flags without legacy flags, return as-is.
  has_global_flag = any(
      arg in {"-h", "--help", "-v", "--version"} for arg in argv
  )
  has_legacy_flag = any(
      arg
      in (
          "-p",
          "--process",
          "--metric",
          "-g",
          "--group",
          "--list-metrics",
          "--list_metrics",
      )
      for arg in argv
  )
  if has_global_flag and not has_legacy_flag:
    return argv

  rewritten = []

  # Check for process flag (-p or --process)
  if "-p" in argv or "--process" in argv:
    print(
        "WARNING: Legacy flag '-p' / '--process' is deprecated. Routing to"
        " 'pmon' subcommand.",
        file=sys.stderr,
    )
    rewritten.append("pmon")
    for arg in argv:
      if arg not in ("-p", "--process"):
        rewritten.append(arg)
    return rewritten

  # Check for list-metrics
  if "--list-metrics" in argv or "--list_metrics" in argv:
    print(
        "WARNING: Legacy flag '--list-metrics' is deprecated. Routing to 'query"
        " --list-metrics'.",
        file=sys.stderr,
    )
    rewritten.append("query")
    for arg in argv:
      if arg in ("--list-metrics", "--list_metrics"):
        rewritten.append("--list-metrics")
      else:
        rewritten.append(arg)
    return rewritten

  # Check for query metric flags
  if "--metric" in argv or "-g" in argv or "--group" in argv:
    print(
        "WARNING: Legacy metric query flags are deprecated. Routing to 'query'"
        " subcommand.",
        file=sys.stderr,
    )
    rewritten.append("query")
    rewritten.extend(argv)
    return rewritten

  # Default fallback if there's no subcommand but flags like --streaming:
  return ["dashboard"] + argv


def parse_arguments(argv: list[str] | None = None):
  """Parses command line arguments for the tpu-info tool."""
  if argv is None:
    argv = sys.argv[1:]

  argv = preprocess_legacy_args(argv)

  # Parent parser for shared options
  parent_parser = argparse.ArgumentParser(add_help=False)
  parent_parser.add_argument(
      "-v",
      "--version",
      action="store_true",
      help="Displays version info.",
  )
  parent_parser.add_argument(
      "--format",
      choices=["table", "json", "csv"],
      default="table",
      help="Output format (default: table).",
  )

  parser = argparse.ArgumentParser(
      description="Display TPU info and metrics.",
      formatter_class=argparse.RawTextHelpFormatter,
      parents=[parent_parser],
  )

  subparsers = parser.add_subparsers(
      dest="subcommand", help="Subcommand to execute"
  )

  # 1. dashboard subcommand
  dash_parser = subparsers.add_parser(
      "dashboard",
      help="Launches the real-time summary dashboard.",
      parents=[parent_parser],
  )
  dash_parser.add_argument(
      "--streaming",
      action="store_true",
      default=False,
      help="Enable streaming mode to refresh metrics continuously",
  )
  dash_parser.add_argument(
      "--no-streaming",
      dest="streaming",
      action="store_false",
      help="Disable streaming mode",
  )
  dash_parser.add_argument(
      "--rate",
      type=float,
      default=1.0,
      help="Refresh rate in seconds for streaming mode.",
  )

  # 2. pmon subcommand
  pmon_parser = subparsers.add_parser(
      "pmon",
      help="Launches a dedicated process and memory monitor.",
      parents=[parent_parser],
  )
  pmon_parser.add_argument(
      "--streaming",
      dest="streaming",
      action="store_true",
      default=False,
      help="Enable streaming mode.",
  )
  pmon_parser.add_argument(
      "--no-streaming",
      dest="streaming",
      action="store_false",
      help="Disable streaming mode.",
  )
  pmon_parser.add_argument(
      "--rate",
      type=float,
      default=1.0,
      help="Refresh rate in seconds.",
  )

  # 3. query subcommand
  query_parser = subparsers.add_parser(
      "query",
      help="Extracts targeted metrics with custom filtering and formatting.",
      parents=[parent_parser],
  )
  query_parser.add_argument(
      "--list-metrics",
      "--list_metrics",
      dest="list_metrics",
      action="store_true",
      help="List all supported metrics.",
  )
  query_parser.add_argument(
      "-g",
      "--group",
      action="append",
      help=(
          "Metric namespace group to display (runtime, execution, network,"
          " orbax, pygrain). Can be specified multiple times."
      ),
  )
  query_parser.add_argument(
      "--metric",
      nargs="+",
      action=_MetricAndFilterAction,
      help=(
          "Metric to display. Can be specified multiple times. Use"
          " --list-metrics to see all supported metrics."
      ),
  )
  query_parser.add_argument(
      "-f",
      "--filter",
      dest="metric",
      action=_MetricAndFilterAction,
      help="Filter string for preceding metric.",
  )

  # 4. topo subcommand
  subparsers.add_parser(
      "topo",
      help="Displays hardware interconnect topology and PCI/IOMMU mappings.",
      parents=[parent_parser],
  )

  return parser.parse_args(argv)
