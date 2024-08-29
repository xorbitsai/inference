# Copyright 2022-2023 XProbe Inc.
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


def test_is_valid_model_name():
    from ...utils import is_valid_model_name

    assert is_valid_model_name("foo")
    assert is_valid_model_name("foo-bar")
    assert is_valid_model_name("foo_bar")
    assert is_valid_model_name("123")
    assert is_valid_model_name("foo@bar")
    assert is_valid_model_name("_foo")
    assert is_valid_model_name("-foo")
    assert not is_valid_model_name("foo bar")
    assert not is_valid_model_name("foo/bar")
    assert not is_valid_model_name("   ")
    assert not is_valid_model_name("")
