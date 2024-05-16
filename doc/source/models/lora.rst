.. _lora:

================
LoRA Integration
================

Currently, Xinference supports launching ``LLM`` and ``image`` models with an attached LoRA fine-tuned model.

Usage
#####

Launch
======
Different from built-in models, xinference currently does not involve managing LoRA models.
Users need to first download the LoRA model themselves and then provide the storage path of the model files to xinference.

.. tabs::

  .. code-tab:: bash shell

    xinference launch <options> 
    --lora-modules <lora_name1> <lora_model_path1>
    --lora-modules <lora_name2> <lora_model_path2>
    --image-lora-load-kwargs <load_params1> <load_value1>
    --image-lora-load-kwargs <load_params2> <load_value2>
    --image-lora-fuse-kwargs <fuse_params1> <fuse_value1>
    --image-lora-fuse-kwargs <fuse_params2> <fuse_value2>

  .. code-tab:: python

    from xinference.client import Client

    client = Client("http://<XINFERENCE_HOST>:<XINFERENCE_PORT>")

    lora_model1={'lora_name': <lora_name1>, 'local_path': <lora_model_path1>}
    lora_model2={'lora_name': <lora_name2>, 'local_path': <lora_model_path2>}
    lora_models=[lora_model1, lora_model2]
    image_lora_load_kwargs={'<load_params1>': <load_value1>, '<load_params2>': <load_value2>},
    image_lora_fuse_kwargs={'<fuse_params1>': <fuse_value1>, '<fuse_params2>': <fuse_value2>}

    peft_model_config = {
    "image_lora_load_kwargs": image_lora_load_params,
    "image_lora_fuse_kwargs": image_lora_fuse_params,
    "lora_list": lora_models
    }
    
    client.launch_model(
        <other_options>,
        peft_model_config=peft_model_config
    )


Apply
=====
For LLM models, you can only configure one lora model you want when you use the model.
Specifically, specify that the ``lora_name`` parameter be configured in the ``generate_config``.
``lora_name`` corresponds to the name of the lora in the LAUNCH procedure described above.

.. tabs::

  .. code-tab:: python

    from xinference.client import Client

    client = Client("http://<XINFERENCE_HOST>:<XINFERENCE_PORT>")
    model = client.get_model("<model_uid>")
    model.chat(
        "<prompt>",
        <other_options>,
        generate_config={"lora_name": "<your_lora_name>"}
    )


Note
####

* The options ``image_lora_load_kwargs`` and ``image_lora_fuse_kwargs`` are only applicable to models with model_type ``image``.
  They correspond to the parameters in the ``load_lora_weights`` and ``fuse_lora`` interfaces of the ``diffusers`` library.
  If launching an LLM model, these parameters are not required.

* You need to add the parameter lora_name during inference to specify the corresponding lora model. You can specify it in the Additional Inputs option.

* For LLM chat models, currently only LoRA models are supported that do not change the prompt style.

* When using GPU, both LoRA and its base model occupy the same devices.
