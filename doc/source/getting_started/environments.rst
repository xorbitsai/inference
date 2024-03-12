.. _environments:

======================
Environments Variables
======================

XINFERENCE_ENDPOINT
~~~~~~~~~~~~~~~~~~~~
Endpoint of Xinference, used to connect to Xinference service.
Default value is http://127.0.0.1:9997 , you can get it through logs.

XINFERENCE_MODEL_SRC
~~~~~~~~~~~~~~~~~~~~~
Modelhub used for downloading models. Default is "huggingface", or you
can set "modelscope" as downloading source.

XINFERENCE_HOME
~~~~~~~~~~~~~~~~
By default, Xinference uses ``<HOME>/.xinference`` as home path to store
necessary files such as logs and models, where ``<HOME>`` is the home
path of current user. You can change this directory by configuring this environment
variable.

XINFERENCE_HEALTH_CHECK_ATTEMPTS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The number of attempts for the health check at Xinference startup, if exceeded,
will result in an error. The default value is 3.

XINFERENCE_HEALTH_CHECK_INTERVAL
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The timeout duration for the health check at Xinference startup, if exceeded,
will result in an error. The default value is 3.

XINFERENCE_DISABLE_HEALTH_CHECK
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Xinference will automatically report health check at Xinference startup.
Setting this environment to 1 can disable health check.

XINFERENCE_DISABLE_VLLM
~~~~~~~~~~~~~~~~~~~~~~~~
Xinference will automatically use vLLM as backend if conditions are met.
Setting this environment to 1 can disable the use of vLLM.