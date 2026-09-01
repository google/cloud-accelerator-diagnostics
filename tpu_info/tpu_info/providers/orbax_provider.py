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

"""Orbax checkpointing metric provider for TPU Info.

This module registers metrics for the Orbax checkpointing framework, mapping
short metric identifiers to Prometheus metric paths and supporting both direct
Prometheus scraping and batch pre-scraped metric extraction.
"""

from collections.abc import Callable, Sequence
from typing import Any

from tpu_info import cli_helper
from tpu_info import metrics
from tpu_info import registry
from rich import console


def _make_orbax_fetcher(
    name: str, path: str
) -> Callable[..., list[console.RenderableType]]:
  """Creates and registers an Orbax metric retrieval handler with the registry.

  Generates a formatted metric title and description combining the metric name
  and its underlying Prometheus path, registers the metric with the
  MetricRegistry under the 'orbax' group, and returns the fetcher callable.

  Args:
    name: The short metric name (e.g., 'orbax_write_size').
    path: The Prometheus metric path/family name.

  Returns:
    A fetcher callable that handles single or batch metric rendering.
  """
  title = cli_helper.format_metric_title(name)
  description = f"{title} ({path})"

  @registry.register_metric(name=name, group="orbax", description=description)
  def fetcher(
      *,
      pre_scraped_metrics: Sequence[metrics.Metric] | None = None,
      skip_if_missing: bool = False,
      **_kwargs: Any,
  ) -> list[console.RenderableType]:
    """Retrieves or extracts Orbax metric tables.

    If pre_scraped_metrics is provided (batch scraping mode), extracts the
    metric from the pre-collected metric families. Otherwise, performs a direct
    Prometheus scrape for the individual metric.

    Args:
      pre_scraped_metrics: Optional sequence of pre-scraped Prometheus metric
        families collected during batch rendering.
      skip_if_missing: If True, suppress table rendering when the metric has no
        active samples in pre_scraped_metrics.
      **_kwargs: Arbitrary keyword arguments forwarded by the registry.

    Returns:
      A list of Rich renderable tables for the Orbax metric.
    """
    if pre_scraped_metrics is not None:
      return cli_helper.get_prometheus_metric_table_from_families(
          pre_scraped_metrics,
          name,
          "Orbax",
          skip_if_missing=skip_if_missing,
      )
    return cli_helper.get_prometheus_metric_table(name)

  return fetcher


for metric_name, prom_path in metrics.ORBAX_SHORT_TO_LONG_MAP.items():
  globals()[f"get_{metric_name}"] = _make_orbax_fetcher(metric_name, prom_path)
