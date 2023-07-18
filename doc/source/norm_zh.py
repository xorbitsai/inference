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

# This file is derived from https://github.com/mars-project/mars/blob/master/docs/source/norm_zh.py

import datetime
import os

from babel.messages import pofile
from babel.messages.pofile import escape


def _zh_len(s):
    """
    Calculate text length in Chinese
    """
    try:
        return len(s.encode("gb2312"))
    except ValueError:
        return len(s)


def _zh_split(s):
    """
    Split text length in Chinese
    """
    import jieba

    try:
        s.encode("ascii")
        has_zh = False
    except ValueError:
        has_zh = True

    if has_zh:
        return list(jieba.cut(s))
    else:
        return pofile.WORD_SEP.split(s)


# code modified from babel.messages.pofile (hash 359ecffca479dfe032d0f7210d5cd8160599c816)
def _normalize(string, prefix="", width=76):
    r"""Convert a string into a format that is appropriate for .po files.
    >>> print(normalize('''Say:
    ...   "hello, world!"
    ... ''', width=None))
    ""
    "Say:\n"
    "  \"hello, world!\"\n"
    >>> print(normalize('''Say:
    ...   "Lorem ipsum dolor sit amet, consectetur adipisicing elit, "
    ... ''', width=32))
    ""
    "Say:\n"
    "  \"Lorem ipsum dolor sit "
    "amet, consectetur adipisicing"
    " elit, \"\n"
    :param string: the string to normalize
    :param prefix: a string that should be prepended to every line
    :param width: the maximum line width; use `None`, 0, or a negative number
                  to completely disable line wrapping
    """

    if width and width > 0:
        prefixlen = _zh_len(prefix)
        lines = []
        for line in string.splitlines(True):
            if _zh_len(escape(line)) + prefixlen > width:
                chunks = _zh_split(line)
                chunks.reverse()
                while chunks:
                    buf = []
                    size = 2
                    while chunks:
                        l = _zh_len(escape(chunks[-1])) - 2 + prefixlen  # noqa: E741
                        if size + l < width:
                            buf.append(chunks.pop())
                            size += l
                        else:
                            if not buf:
                                # handle long chunks by putting them on a
                                # separate line
                                buf.append(chunks.pop())
                            break
                    lines.append("".join(buf))
            else:
                lines.append(line)
    else:
        lines = string.splitlines(True)

    if len(lines) <= 1:
        return escape(string)

    # Remove empty trailing line
    if lines and not lines[-1]:
        del lines[-1]
        lines[-1] += "\n"
    return '""\n' + "\n".join([(prefix + escape(line)) for line in lines])


def main():
    try:
        import jieba  # noqa: F401
    except ImportError:
        return

    pofile.normalize = _normalize
    for root, _dirs, files in os.walk("."):
        if "zh" not in root:
            continue
        for f in files:
            if not f.endswith(".po"):
                continue
            path = os.path.join(root, f)

            # only modify recent-changed files
            modify_time = datetime.datetime.fromtimestamp(os.path.getmtime(path))
            if (datetime.datetime.now() - modify_time).total_seconds() > 120:
                continue

            with open(path, "rb") as inpf:
                catalog = pofile.read_po(inpf)
            with open(path, "wb") as outf:
                pofile.write_po(outf, catalog)


if __name__ == "__main__":
    main()
