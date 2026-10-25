# Ming-Image inference code

Source: https://github.com/inclusionAI/Ming-Image

Commit: `62c6072e1ff15af83f7c4963a0a1954c1424e80e`

The upstream repository uses MIT (`LICENSE`). Some source files carry their own
Apache-2.0 headers; those notices are retained, with the license text in
`LICENSE-APACHE`.

This copy contains the Python inference modules required for Xinference. Imports
of modules within this package were changed to relative imports. Qwen RMSNorm
uses the equivalent PyTorch implementation, so Transformer Engine is not needed.
