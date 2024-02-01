.. _tools:

=====================
Tools
=====================

Learn how to connect LLM with external tools.


Introduction
============

With the ``tools`` ability you can have your model use external tools. 


Like `OpenAI's Function calling API <https://platform.openai.com/docs/guides/function-calling>`_, you can define the functions along
with their parameters and have the model dynamically choose which function to call and what parameters to pass to it.

This is the general process for calling a function:

1. You submit a query, detailing the functions, their parameters, and descriptions.
2. The LLM decides whether to initiate the function. If chosen not to, it replies in everyday language,
   either offering a solution based on its inherent understanding or asking further details about the query
   and tool usage. On deciding to use a tool, it recommends the suitable API and instructions for its usage, framed in JSON.
3. Following that, you implement the API call within your application and send the returned response back to the LLM
   for result analysis and proceeding with the next steps.

There is no dedicated API endpoint implemented for ``tools`` ability. It must be used in combination with Chat API.
  
Supported models
-------------------

The ``tools`` ability is supported with the following models in Xinference:

* :ref:`models_llm_qwen-chat`
* :ref:`models_llm_chatglm3`
* :ref:`models_llm_gorilla-openfunctions-v1`


Quickstart
==============

An optional parameter ``tools`` in the Chat API can be used to provide function specifications.
The purpose of this is to enable models to generate function arguments which adhere to the provided specifications. 

Example using OpenAI Client
------------------------------

.. code-block::

    import openai

    client = openai.Client(
        api_key="cannot be empty", 
        base_url="http://<XINFERENCE_HOST>:<XINFERENCE_PORT>/v1"
    )
    client.chat.completions.create(
        model="<MODEL_UID>",
        messages=[{
            "role": "user",
            "content": "Call me an Uber ride type 'Plus' in Berkeley at zipcode 94704 in 10 minutes"
        }],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "uber_ride",
                    "description": "Find suitable ride for customers given the location, "
                    "type of ride, and the amount of time the customer is "
                    "willing to wait as parameters",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "loc": {
                                "type": "int",
                                "description": "Location of the starting place of the Uber ride",
                            },
                            "type": {
                                "type": "string",
                                "enum": ["plus", "comfort", "black"],
                                "description": "Types of Uber ride user is ordering",
                            },
                            "time": {
                                "type": "int",
                                "description": "The amount of time in minutes the customer is willing to wait",
                            },
                        },
                    },
                },
            }
        ],
    )
    print(response.choices[0].message)


The output will be:

.. code-block:: json

  {
      "role": "assistant",
      "content": null,
      "tool_calls": [
          "id": "call_ad2f383f-31c7-47d9-87b7-3abe928e629c", 
          "type": "function", 
          "function": {
              "name": "uber_ride", 
              "arguments": "{\"loc\": 94704, \"type\": \"plus\", \"time\": 10}"
          }
      ],
  }

.. note::

  Finish reason will be ``tool_calls`` if the LLM uses a tool call. Othewise it will be the default finish reason.


.. note::

  The API will not actually execute any function calls. It is up to developers to execute function calls using model outputs.



You can find more examples of ``tools`` ability in the tutorial notebook:

.. grid:: 1

   .. grid-item-card:: Function calling
      :link: https://github.com/xorbitsai/inference/blob/main/examples/FunctionCall.ipynb
      
      Learn from a complete example demonstrating function calling

