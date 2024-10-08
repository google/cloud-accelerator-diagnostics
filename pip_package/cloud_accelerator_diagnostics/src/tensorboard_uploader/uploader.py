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

"""Uploader module.

This module provides the functionality to upload data to Tensorboard in Vertex
AI.
"""

import logging

from cloud_accelerator_diagnostics.pip_package.cloud_accelerator_diagnostics.src.tensorboard_uploader import tensorboard
from google.cloud.aiplatform import aiplatform


logger = logging.getLogger(__name__)


def start_upload_to_tensorboard(
    project,
    location,
    experiment_name,
    tensorboard_name,
    logdir,
):
  """Continues to listen for new data in the logdir and uploads when it appears.

  Note that after calling `start_upload_to_tensorboard()`, thread will be kept
  alive even if an exception is thrown. To ensure the thread gets shut down, put
  any code after `start_upload_to_tensorboard()` and before
  `stop_upload_to_tensorboard()` in a `try` statement, and call
  `stop_upload_to_tensorboard()` in finally.

  Sample usage:
  ```
    start_upload_to_tensorboard(project='test-project'
                                location='us-central1',
                                experiment_name='test-experiment',
                                tensorboard_name='test-instance',
                                logdir='test-logdir')
    try:
      # your code here
    finally:
      stop_upload_to_tensorboard()
  ```

  Args:
      project (str): Google Cloud Project that has the Tensorboard instance.
      location (str): Location where Tensorboard instance is present.
      experiment_name (str): The name of the Tensorboard experiment.
      tensorboard_name (str): The name of the Tensorboard instance.
      logdir (str): path of the log directory to upload to Tensorboard.
  """
  try:
    aiplatform.init(project=project, location=location)

    # Skip uploading logs to VertexAI if a Tensorboard instance doesn't exist
    tensorboard_identifiers = tensorboard.get_instance_identifiers(
        tensorboard_name
    )
    if not tensorboard_identifiers:
      logger.error(
          "No Tensorboard instance with the name %s present in the project %s."
          " Skipping uploading logs to VertexAI.",
          tensorboard_name,
          project,
      )
      return
    else:
      # get the first Tensorboard instance even if multiple instances exist
      tensorboard_id = tensorboard_identifiers[0]

    # Skip uploading logs to VertexAI if a Tensorboard experiment doesn't exist
    experiment = tensorboard.get_experiment(tensorboard_id, experiment_name)
    if experiment is None:
      logger.error(
          "No Tensorboard experiment with the name %s present in the project"
          " %s. Skipping uploading logs to VertexAI.",
          experiment_name,
          project,
      )
      return

    start_upload(tensorboard_id, experiment_name, logdir)
  except (ValueError, Exception) as e:
    logger.exception(
        "Error while uploading logs to Tensorboard. This will not impact the"
        " workload. Error: %s",
        e,
    )


def stop_upload_to_tensorboard():
  """Stops the thread created by `start_upload_to_tensorboard()`."""
  logger.info("Logs will no longer be uploaded to Tensorboard.")
  aiplatform.end_upload_tb_log()


def start_upload(tensorboard_id, experiment_name, logdir):
  """Starts uploading logs to Tensorboard instance in VertexAI.

  Args:
    tensorboard_id (str): The id of Tensorboard instance.
    experiment_name (str): The name of the Tensorboard experiment.
    logdir (str): path of the log directory to upload to Tensorboard.
  """
  logger.info("Starting uploading of logs to Tensorboard.")
  try:
    aiplatform.start_upload_tb_log(
        tensorboard_id=tensorboard_id,
        tensorboard_experiment_name=experiment_name,
        logdir=logdir,
    )
  except Exception as e:
    logger.exception(
        "Error while uploading logs to Tensorboard. This will not impact the"
        " workload. Error: %s",
        e,
    )
