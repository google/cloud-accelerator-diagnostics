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

"""Network telemetry metric provider for TPU devices.

This module defines and registers network and interconnect metrics, including
Data Center Network (DCN) transfer latencies, host-to-device transfers,
collective communication latencies, and gRPC TCP metrics with the central
MetricRegistry.
"""

from typing import Any

from tpu_info import cli_helper
from tpu_info.registry import register_metric
from rich import console


def _create_latency_handler(metric_name: str):
  """Creates a latency metric handler closure for the given metric name.

  Args:
    metric_name: The identifier of the network latency metric to handle (e.g.,
      'buffer_transfer_latency', 'grpc_tcp_min_rtt').

  Returns:
    A callable handler that accepts metric filter parameters and returns Rich
    renderable tables from TransferLatencyTables.
  """

  def handler(
      *, filters: dict[str, Any] | None = None, **_kwargs: Any
  ) -> list[console.RenderableType]:
    """Renders transfer latency table for the metric.

    Args:
      filters: Optional dictionary of query filters (e.g. {'percentile':
        'p99'}).
      **_kwargs: Arbitrary keyword arguments forwarded by the registry.

    Returns:
      A list containing the rendered Rich table for the latency metric.
    """
    return [cli_helper.TransferLatencyTables().render(metric_name, filters)]

  return handler


_NETWORK_METRICS = {
    "buffer_transfer_latency": (
        "Cumulative distribution of DCN transfer latencies in microseconds",
        {"percentile"},
    ),
    "inbound_buffer_transfer_latency": (
        (
            "Cumulative distribution of DCN inbound transfer latencies in"
            " microseconds"
        ),
        {"percentile"},
    ),
    "host_to_device_transfer_latency": (
        (
            "Cumulative distribution of host-to-device transfer latencies in"
            " microseconds"
        ),
        {"percentile"},
    ),
    "device_to_host_transfer_latency": (
        (
            "Cumulative distribution of device-to-host transfer latencies in"
            " microseconds"
        ),
        {"percentile"},
    ),
    "collective_e2e_latency": (
        (
            "Cumulative distribution of collective end-to-end latencies in"
            " microseconds"
        ),
        {"percentile"},
    ),
    "host_compute_latency": (
        (
            "Cumulative distribution of MXLA compute latencies on host in"
            " microseconds"
        ),
        {"percentile"},
    ),
    "grpc_tcp_min_rtt": (
        (
            "Cumulative distribution of gRPC TCP minimum round-trip time in"
            " microseconds"
        ),
        None,
    ),
    "grpc_tcp_delivery_rate": (
        "Cumulative distribution of gRPC TCP delivery rate in Mbps",
        None,
    ),
}

for _name, (_desc, _filters) in _NETWORK_METRICS.items():
  register_metric(_name, "network", _desc, allowed_filters=_filters)(
      _create_latency_handler(_name)
  )
