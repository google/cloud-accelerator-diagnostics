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

"""Provider package for TPU telemetry metrics.

This package serves as the entry point and registry initializer for domain
metric providers. Importing this package loads all domain-specific provider
modules (runtime, execution, network, orbax, pygrain), triggering metric
registration decorators with the central MetricRegistry.
"""

from tpu_info.providers import execution_provider
from tpu_info.providers import network_provider
from tpu_info.providers import orbax_provider
from tpu_info.providers import pygrain_provider
from tpu_info.providers import runtime_provider

__all__ = [
    "execution_provider",
    "network_provider",
    "orbax_provider",
    "pygrain_provider",
    "runtime_provider",
]

