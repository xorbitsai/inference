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

from ..utils import (
    build_replica_model_uid,
    is_valid_model_uid,
    iter_replica_model_uid,
    parse_replica_model_uid,
)


def test_replica_model_uid():
    all_gen_ids = []
    for replica_model_uid in iter_replica_model_uid("abc", 5):
        rebuild_replica_model_uid = build_replica_model_uid(
            *parse_replica_model_uid(replica_model_uid)
        )
        assert rebuild_replica_model_uid == replica_model_uid
        all_gen_ids.append(replica_model_uid)
    assert len(all_gen_ids) == 5
    assert len(set(all_gen_ids)) == 5


def test_is_valid_model_uid():
    from ...client.restful.restful_client import Client

    assert is_valid_model_uid("foo")
    assert is_valid_model_uid("foo-bar")
    assert is_valid_model_uid("foo_bar")
    assert is_valid_model_uid("123")
    assert not is_valid_model_uid("foo@bar")
    assert not is_valid_model_uid("foo bar")
    assert not is_valid_model_uid("_foo")
    assert not is_valid_model_uid("-foo")
    for _ in range(10):
        assert is_valid_model_uid(Client._gen_model_uid())
