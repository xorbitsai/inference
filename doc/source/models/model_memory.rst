.. _cal_model_memory:

========================
Model Memory Calculation
========================

For better planning of VMEM usage, xinference provided tool for model memory calculation: ``cal-model-mem``

Use algorithm from https://github.com/RahulSChand/gpu_poor

Output: model_mem, kv_cache, overhead, active_mem

Example: 
To calculate memory usage for qwen1.5-chat, run the following command:

.. tabs::

  .. code-tab:: bash Command
    
    xinference cal-model-mem -s 7 -q Int4 -f gptq -c 16384 -n qwen1.5-chat
    
  .. code-tab:: bash Output
    
    model_name: qwen1.5-chat
    kv_cache_dtype: 16
    model size: 7.0 B
    quant: Int4
    context: 16384
    gpu mem usage:
      model mem: 4139 MB
      kv_cache: 8192 MB
      overhead: 650 MB
      active: 17024 MB
      total: 30005 MB (30 GB)

Syntax
------

* --size-in-billions {model_size}
  
  
  * -s {model_size}


  Set the model size.
  Specify the model size in billions of parameters. Format accept 1_8 and 1.8.
  For example, 7 for 7.0B model size.


* --quantization {precision}
  
  
  * -q {precision} *(Optional)*


  Define the quantization settings for the model.
  For example, Int4 for INT4 quantization.


* --model-name {model_name}
  
  
  * -n {model_name} *(Optional)*


  Specify the model's name.
  If provided, fetch model config from huggingface/modelscope; If not specified, use default model layer to estimate.

 
* --context-length {context_length}
  
  
  * -c {context_length}


  Specify the maximum number of tokens(context length) that your model support.

 
* --model-format {format}
  
  
  * -f {format}


  Specify the format of the model, e.g. pytorch, ggmlv3, etc.

 
.. note::
  The environment variable ``HF_ENDPOINT`` could set the endpoint of HuggingFace. e.g. hf-mirror, etc.
  Please refer to :ref:`this document <models_download>`

