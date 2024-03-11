.. _lora:

================
LoRA Integration
================

Currently, Xinference supports launching ``LLM`` and ``image`` models with an attached LoRA fine-tuned model.

Usage
^^^^^
Different from built-in models, xinference currently does not involve managing LoRA models.
Users need to first download the LoRA model themselves and then provide the storage path of the model files to xinference.

.. tabs::

  .. code-tab:: bash shell

    xinference launch <options> --peft-model-path <lora_model_path>
    --image-lora-load-kwargs <load_params1> <load_value1>
    --image-lora-load-kwargs <load_params2> <load_value2>
    --image-lora-fuse-kwargs <fuse_params1> <fuse_value1>
    --image-lora-fuse-kwargs <fuse_params2> <fuse_value2>

  .. code-tab:: python

    from xinference.client import Client

    client = Client("http://<XINFERENCE_HOST>:<XINFERENCE_PORT>")
    client.launch_model(
        <other_options>,
        peft_model_path='<lora_model_path>',
        image_lora_load_kwargs={'<load_params1>': <load_value1>, '<load_params2>': <load_value2>},
        image_lora_fuse_kwargs={'<fuse_params1>': <fuse_value1>, '<fuse_params2>': <fuse_value2>}
    )


Note
^^^^

* The options ``image_lora_load_kwargs`` and ``image_lora_fuse_kwargs`` are only applicable to models with model_type ``image``.
  They correspond to the parameters in the ``load_lora_weights`` and ``fuse_lora`` interfaces of the ``diffusers`` library.
  If launching an LLM model, these parameters are not required.

* For LLM chat models, currently only LoRA models are supported that do not change the prompt style.

* When using GPU, both LoRA and its base model occupy the same devices.
