.. _examples_ai_podcast:

======================
Example: AI Podcast 🎙
======================

**Description**:

🎙️AI Podcast - Voice Conversations with Multiple Agents on M2 Max 💻

**Support Language** :

English (AI_Podcast.py)

Chinese (AI_Podcast_ZH.py)

**Used Technology (EN version)** :

    @ `OpenAI <https://twitter.com/OpenAI>`_ 's `whisper <https://pypi.org/project/openai-whisper/>`_

    @ `ggerganov <https://twitter.com/ggerganov>`_ 's `ggml <https://github.com/ggerganov/ggml>`_

    @ `WizardLM_AI <https://twitter.com/WizardLM_AI>`_ 's `wizardlm v1.0 <https://huggingface.co/WizardLM>`_

    @ `lmsysorg <https://twitter.com/lmsysorg>`_ 's `vicuna v1.3 <https://huggingface.co/lmsys/vicuna-7b-v1.3>`_

    @ `Xinference <https://github.com/xorbitsai/inference>`_ as a launcher

**Detailed Explanation on the Demo Functionality** :

1. Generate the Wizardlm Model and Vicuna Model when the program is launching with Xorbits Inference.
   Initiate the Chatroom by giving the two chatbot their names and telling them that there is a human user
   called "username", where "username" is given by user's input. Initialize a empty chat history for the chatroom.

2. Use Audio device to store recording into file, and transcribe the file using OpenAI's Whisper to receive a human readable text as string.

3. Based on the input message string, determine which agents the user want to talk to. Call the target agents and
   parse in the input string and chat history for the model to generate.

4. When the responses are ready, use Macos's "Say" Command to produce audio through speaker. Each agents have their
   own voice while speaking.

5. Store the user input and the agent response into chat history, and recursively looping the program until user
   explicitly says words like "see you" in their responses.

**Highlight Features with Xinference** :

1. With Xinference's distributed system, we can easily deploy two different models in the same session and in the
   same "chatroom". With enough resources, the framework can deploy any amount of models you like at the same time.

2. With Xinference, you can deploy the model easily by just adding a few lines of code.
   For examples, for launching the vicuna model in the demo, just by::

     args = parser.parse_args()
     endpoint = args.endpoint
     client = Client(endpoint)

     model_a = "vicuna-v1.3"
     model_a_uid = client.launch_model(
         model_name=model_a,
         model_format="ggmlv3",
         model_size_in_billions=7,
         quantization="q4_0",
         n_ctx=2048,
     )
     model_a_ref = client.get_model(model_a_uid)

   Then, the Xinference client will handle "target model downloading and caching", "set up environment and process
   for the model", and "run the service at selected endpoint. " You are now ready to play with your llm model.

**Original Demo Video** :

    * `🎙️AI Podcast - Voice Conversations with Multiple Agents on M2 Max💻🔥🤖 <https://twitter.com/yichaocheng/status/1679129417778442240>`_

**Source Code** :

    * `AI_Podcast <https://github.com/xorbitsai/inference/blob/main/examples/AI_podcast.py>`_ (English Version)

    * AI_Podcast_ZH (Chinese Version)