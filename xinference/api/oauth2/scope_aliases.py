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
"""Scope alias bridge for the permission/scope alignment migration.

Tokens issued before the scope-rename alignment carry legacy scope names
(``models:start``, ``models:stop``, ``models:add``, ``models:unregister``)
that no longer match any route's ``Security(scopes=[...])`` declaration.
``SCOPE_ALIASES`` maps those legacy names to their current equivalents so
``_normalize_scopes`` can expand them before the scope-by-scope check.

Scheduled for removal in a future release once deprecation warnings
confirm zero alias hits in production.
"""

from typing import Iterable, Optional, Set

SCOPE_ALIASES = {
    "models:start": "models:write",
    "models:stop": "models:write",
    "models:add": "models:register",
    "models:unregister": "models:register",
}


def _normalize_scopes(scopes: Optional[Iterable[str]]) -> Set[str]:
    """Expand legacy scope aliases so old tokens work on renamed routes.

    Returns a set that includes both the original scopes and their
    alias-mapped equivalents. A token carrying ``models:start`` therefore
    passes a ``models:write`` scope check (and vice versa — though new
    tokens should only carry the new names).
    """
    expanded: Set[str] = set(scopes or [])
    for s in scopes or []:
        if s in SCOPE_ALIASES:
            expanded.add(SCOPE_ALIASES[s])
    return expanded
