# ControlNet preprocessing sources

Imported from `xorbitsai/xinference-backend` commit
`a979c575` (`xinference/thirdparty/controlnet`), originally introduced by
`f6b43017` (full SDAPI support, PR #68).

This is vendored third-party code, not a new Apache-2.0 implementation.
The preprocessor registry and Forge compatibility code originate from
<https://github.com/lllyasviel/stable-diffusion-webui-forge> (AGPL-3.0;
see LICENSE.forge.txt). ControlNet annotator integrations also originate from
<https://github.com/Mikubill/sd-webui-controlnet> (GPL-3.0;
see LICENSE.controlnet.txt). Nested annotator directories retain their own
LICENSE files and copyright notices. The image-layer prompt weighting and
random-noise compatibility helpers also derive from the WebUI ecosystem.

Local adaptations defer MobileSAM and FaceXLib imports until the selected
preprocessor is used. Model weights are downloaded on demand and are not
included in this directory. Weight licenses are separate from source licenses.
