# Copyright 2022-2026 Xinference Holdings Pte. Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class ModelNotReadyError(Exception):
    """Raised when a model is still loading and cannot serve requests yet."""

    pass


class InvalidModelRequestError(ValueError):
    """Raised by explicit model validators for invalid request parameters."""

    @property
    def client_message(self) -> str:
        """Return the validation message without xoscar's actor prefix."""
        if self.args:
            return str(self.args[0])
        return "Invalid model request."


class InvalidAudioInputError(ValueError):
    """Raised when client-provided audio cannot be used by a model."""

    @property
    def client_message(self) -> str:
        """Return a stable message without xoscar's actor diagnostic prefix."""
        if self.args:
            return str(self.args[0])
        return "Invalid audio input."


class ModelCatalogUpdateError(RuntimeError):
    """A worker could not download or store a model catalog update."""


class ModelCatalogUnavailableError(ModelCatalogUpdateError):
    """No workers are available to refresh the model catalog."""
