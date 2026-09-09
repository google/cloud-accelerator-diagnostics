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

"""Central registry and descriptor for telemetry metrics."""

from collections.abc import Callable
import dataclasses
import importlib
import os
import sys
import threading
from typing import Any

from tpu_info import device
from rich import console
from rich import panel
from rich import text


@dataclasses.dataclass
class MetricDescriptor:
  """Describes a telemetry metric and its retrieval handler.

  Attributes:
    name: Unique identifier for the metric.
    group: Functional category or domain grouping for the metric.
    description: Human-readable summary of what the metric measures.
    handler: Callable handler function that retrieves and formats metric data
      into Rich renderables.
    allowed_filters: Optional set of supported filter key names accepted by the
      metric handler.
  """

  name: str
  group: str
  description: str
  handler: Callable[..., list[console.RenderableType]]
  allowed_filters: set[str] | None = None


class MetricRegistry:
  """Singleton registry for telemetry metrics and render handlers.

  Maintains registered metric descriptors across domain providers and provides
  lookup, grouping, filtering schema extraction, and metric rendering
  dispatching.
  """

  _instance: "MetricRegistry |  None" = None
  _lock: threading.RLock = threading.RLock()
  _descriptors: dict[str, MetricDescriptor]
  _loaded: bool

  def __new__(cls) -> "MetricRegistry":
    """Either constructs or returns the singleton instance of MetricRegistry.

    Returns:
      The singleton MetricRegistry instance.
    """
    with cls._lock:
      if cls._instance is None:
        instance = super().__new__(cls)
        instance._descriptors = {}
        instance._loaded = False
        cls._instance = instance
    return cls._instance

  @classmethod
  def _reset(cls) -> None:
    """Resets the singleton instance (primarily for testing)."""
    with cls._lock:
      cls._instance = None

  def _ensure_providers_loaded(self) -> None:
    """Ensures that all domain metric provider modules are imported.

    Dynamically imports the providers package once using double-checked locking
    to register all built-in metrics with this registry.
    """
    if not self._loaded:
      with self._lock:
        # Prevent race conditions if multiple threads call this method.
        if not self._loaded:
          package_name_full = (
              __package__
              or "tpu_info"  # Fallback for open-source layout.
          )
          provider_prefix = f"{package_name_full}.providers"
          try:
            for mod_key in list(sys.modules.copy()):
              if mod_key == provider_prefix or mod_key.startswith(
                  provider_prefix + "."
              ):
                sys.modules.pop(mod_key, None)
            importlib.import_module(provider_prefix)
            self._loaded = True
          except (ImportError, ModuleNotFoundError) as e:
            if getattr(e, "name", None) == provider_prefix:
              self._loaded = True
            else:
              raise

  def register(
      self,
      name: str,
      group: str,
      description: str,
      allowed_filters: set[str] | None = None,
  ) -> Callable[
      [Callable[..., list[console.RenderableType]]],
      Callable[..., list[console.RenderableType]],
  ]:
    """Decorator to register a metric retrieval handler with the registry.

    Args:
      name: Unique identifier for the metric.
      group: Domain group or subsystem name the metric belongs to.
      description: Human-readable summary describing the metric.
      allowed_filters: Optional set of allowed filter parameter names accepted
        by the metric handler.

    Returns:
      A decorator function that takes the metric handler, registers it as a
      MetricDescriptor, and returns the original handler.
    """

    def decorator(
        handler: Callable[..., list[console.RenderableType]],
    ) -> Callable[..., list[console.RenderableType]]:
      descriptor = MetricDescriptor(
          name=name,
          group=group,
          description=description,
          handler=handler,
          allowed_filters=allowed_filters,
      )
      self._descriptors[name] = descriptor
      return handler

    return decorator

  def get_descriptor(self, name: str) -> MetricDescriptor | None:
    """Retrieves the metric descriptor for a given metric name.

    Args:
      name: The unique identifier of the metric to look up.

    Returns:
      The MetricDescriptor instance corresponding to the name, or None if the
      metric is not registered.
    """
    self._ensure_providers_loaded()
    return self._descriptors.get(name)

  def get_all_descriptors(self) -> dict[str, MetricDescriptor]:
    """Returns a shallow copy of all registered metric descriptors.

    Returns:
      A dictionary mapping metric name strings to their corresponding
      MetricDescriptor instances.
    """
    self._ensure_providers_loaded()
    return dict(self._descriptors)

  def get_all_metric_names(self) -> frozenset[str]:
    """Returns the names of all registered metrics.

    Returns:
      An immutable frozenset containing all registered metric name strings.
    """
    self._ensure_providers_loaded()
    return frozenset(self._descriptors.keys())

  def get_all_filter_schemas(self) -> dict[str, set[str]]:
    """Returns filter schemas for all metrics that support filtering.

    Returns:
      A dictionary mapping metric names to the set of allowed filter parameter
      names for each metric that defines filter schemas.
    """
    self._ensure_providers_loaded()
    return {
        d.name: d.allowed_filters
        for d in self._descriptors.values()
        if d.allowed_filters is not None
    }

  def get_descriptors_by_group(self, group: str) -> list[MetricDescriptor]:
    """Retrieves all metric descriptors belonging to a specific group.

    Args:
      group: The domain group name to filter metric descriptors by (e.g.
        "runtime", "execution", "network", "orbax", "pygrain").

    Returns:
      A list of MetricDescriptor instances registered under the given group.
    """
    self._ensure_providers_loaded()
    return [d for d in self._descriptors.values() if d.group == group]

  def get_all_groups(self) -> list[str]:
    """Returns a sorted list of all unique metric group names registered.

    Returns:
      A sorted list of unique group name strings in alphabetical order.
    """
    self._ensure_providers_loaded()
    return sorted(list(set(d.group for d in self._descriptors.values())))

  def _batch_scrape_prometheus(
      self,
      batch_metrics: list[tuple[str, dict[str, Any] | None]],
      telemetry_name: str,
      default_port: int,
      port_env_var: str,
      env_var_name: str,
      chip_type: Any = None,
      count: int = 0,
  ) -> list[console.RenderableType]:
    """Scrapes Prometheus once for a batch of metrics and renders their tables.

    This method optimizes metric collection by executing a single Prometheus
    scrape per telemetry source (e.g. Orbax, Pygrain) for multiple requested
    metrics rather than scraping the endpoint repeatedly. It forwards the
    pre-scraped metric families to each metric descriptor's handler, handles
    offline Prometheus connection errors gracefully by returning a warning
    panel, and consolidates missing metric warnings when multiple metrics are
    queried in a single batch.

    Args:
      batch_metrics: A list of tuples containing metric names and optional
        filter dictionaries to query.
      telemetry_name: Display name of the telemetry source (e.g., "Orbax",
        "Pygrain").
      default_port: Default port number to use if the port environment variable
        is unset or invalid.
      port_env_var: Name of the environment variable containing the Prometheus
        server port.
      env_var_name: Name of the workload environment variable needed to enable
        telemetry (used in connection error instructions).
      chip_type: Optional TPU chip type passed to individual metric handlers.
      count: Number of chips/cores passed to individual metric handlers.

    Returns:
      A list of Rich renderable tables or warning/error panels.
    """
    from tpu_info import metrics  # pylint: disable=g-import-not-at-top

    port_str = os.environ.get(port_env_var)
    if port_str:
      try:
        port = int(port_str)
      except ValueError:
        port = default_port
    else:
      port = default_port

    try:
      all_metrics = metrics.scrape_prometheus(port)
    except metrics.PrometheusConnectionError:
      warning_message = (
          f"Could not connect to {telemetry_name} Prometheus server on port"
          f" {port}.\nEnsure your workload is running with"
          f" [bold]{env_var_name}[/bold] environment variable."
      )
      return [
          panel.Panel(
              warning_message,
              title=(
                  f"[bold yellow]{telemetry_name} Telemetry Offline[/bold"
                  " yellow]"
              ),
              border_style="yellow",
          )
      ]
    except Exception as e:  # pylint: disable=broad-exception-caught
      return [
          panel.Panel(
              text.Text(f"Error fetching metrics: {e!r}"),
              title="[bold red]Error[/bold red]",
              border_style="red",
          )
      ]

    renderables = []
    missing_metrics = []
    consolidate_warnings = len(batch_metrics) > 1

    for metric_name, filters in batch_metrics:
      descriptor = self.get_descriptor(metric_name)
      if not descriptor or not descriptor.handler:
        continue
      try:
        tables = descriptor.handler(
            pre_scraped_metrics=all_metrics,
            skip_if_missing=consolidate_warnings,
            filters=filters,
            chip_type=chip_type,
            count=count,
        )
      except Exception as e:  # pylint: disable=broad-exception-caught
        renderables.append(
            panel.Panel(
                text.Text(f"Error rendering metric '{metric_name}': {e!r}"),
                title="[bold red]Error[/bold red]",
                border_style="red",
            )
        )
        continue
      if not tables and consolidate_warnings:
        missing_metrics.append(metric_name)
      else:
        renderables.extend(tables)

    if missing_metrics:
      missing_list = "\n".join(f"- {m}" for m in missing_metrics)
      warning_panel = panel.Panel(
          "The following metrics were not found on the Prometheus server"
          f" (they may not have been recorded yet):\n{missing_list}",
          title=(
              f"[bold yellow]{telemetry_name} Metrics Not Found[/bold"
              " yellow]"
          ),
          border_style="yellow",
      )
      renderables.append(warning_panel)

    return renderables

  def render_metrics(
      self,
      validated_metrics: list[tuple[str, dict[str, Any] | None]],
      chip_type: device.TpuChip | None,
      count: int,
  ) -> list[console.RenderableType]:
    """Renders visual tables and panels for validated metrics.

    Dispatches rendering requests to individual metric handlers or groups
    Prometheus-based metrics (Orbax, Pygrain) to batch-scrape their endpoints.
    For metrics that are not found or handlers that encounter issues, returns
    appropriate error panels.

    Args:
      validated_metrics: A list of tuples containing validated metric name
        strings and optional filter dictionaries.
      chip_type: The TPU chip architecture type (e.g. TPU v4, v5e), or None if
        unavailable.
      count: The number of TPU chips or cores present on the host.

    Returns:
      A list of Rich renderables (e.g., Table, Panel, Text) representing the
      formatted metric outputs or error/warning panels.
    """
    self._ensure_providers_loaded()
    renderables: list[console.RenderableType] = []

    orbax_metrics = []
    pygrain_metrics = []
    other_metrics = []

    for metric_name, filters in validated_metrics:
      descriptor = self.get_descriptor(metric_name)
      if not descriptor:
        renderables.append(
            panel.Panel(f"Unknown metric: {metric_name}", border_style="red")
        )
        continue
      if descriptor.group == "orbax":
        orbax_metrics.append((metric_name, filters))
      elif descriptor.group == "pygrain":
        pygrain_metrics.append((metric_name, filters))
      else:
        other_metrics.append((metric_name, filters))

    if orbax_metrics:
      from tpu_info import metrics  # pylint: disable=g-import-not-at-top

      renderables.extend(
          self._batch_scrape_prometheus(
              orbax_metrics,
              telemetry_name="Orbax",
              default_port=metrics.ORBAX_PROMETHEUS_DEFAULT_PORT,
              port_env_var="ORBAX_PROMETHEUS_PORT",
              env_var_name="ENABLE_ORBAX_PROMETHEUS_TELEMETRY=true",
              chip_type=chip_type,
              count=count,
          )
      )

    if pygrain_metrics:
      from tpu_info import metrics  # pylint: disable=g-import-not-at-top

      renderables.extend(
          self._batch_scrape_prometheus(
              pygrain_metrics,
              telemetry_name="Pygrain",
              default_port=metrics.PYGRAIN_PROMETHEUS_DEFAULT_PORT,
              port_env_var="PYGRAIN_PROMETHEUS_PORT",
              env_var_name="ENABLE_PYGRAIN_PROMETHEUS_TELEMETRY=true",
              chip_type=chip_type,
              count=count,
          )
      )

    for metric_name, filters in other_metrics:
      descriptor = self.get_descriptor(metric_name)
      if descriptor and descriptor.handler:
        try:
          renderables.extend(
              descriptor.handler(
                  chip_type=chip_type, count=count, filters=filters
              )
          )
        except Exception as e:  # pylint: disable=broad-exception-caught
          renderables.append(
              panel.Panel(
                  text.Text(f"Error rendering metric '{metric_name}': {e!r}"),
                  title="[bold red]Error[/bold red]",
                  border_style="red",
              )
          )

    return renderables


def register_metric(
    name: str,
    group: str,
    description: str,
    allowed_filters: set[str] | None = None,
) -> Callable[
    [Callable[..., list[console.RenderableType]]],
    Callable[..., list[console.RenderableType]],
]:
  """Convenience decorator delegating to the MetricRegistry singleton.

  Args:
    name: Unique identifier for the metric.
    group: Domain group or subsystem name the metric belongs to.
    description: Human-readable summary describing the metric.
    allowed_filters: Optional set of allowed filter parameter names accepted by
      the metric handler.

  Returns:
    A decorator function that registers the handler with the MetricRegistry
    singleton and returns the original handler function.
  """
  return MetricRegistry().register(
      name, group, description, allowed_filters
  )
