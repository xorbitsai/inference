# Ming-3.0-tts inference runtime

Source: https://github.com/inclusionAI/Ming-3.0-tts
Revision: e999c39fa676722aaa9ea29b5aa19a90525b271f
License: MIT (see LICENSE). Original per-file notices are preserved.

Includes the 0.5B inference modules and text helpers, and the Tiny runtime.
Training, demos, CLI scripts, checkpoints, and optional FST normalization
grammars are excluded. Xinference uses `text_normalization=False` and
`cache_prompts=False` for 0.5B: normalize numbers before synthesis, and avoid
caching request-scoped reference paths and GPU tensors. Tiny keeps the public
release API's numerical defaults and task validation.

Local changes: 0.5B absolute sibling imports use relative package imports;
trailing whitespace is normalized without changing the Python AST.
No global sys.path additions or top-level module aliases are needed.
