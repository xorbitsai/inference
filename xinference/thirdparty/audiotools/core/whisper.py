import torch


class WhisperMixin:
    is_initialized = False

    def setup_whisper(
        self,
        pretrained_model_name_or_path: str = "openai/whisper-base.en",
        device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        from transformers import WhisperForConditionalGeneration
        from transformers import WhisperProcessor

        self.whisper_device = device
        self.whisper_processor = WhisperProcessor.from_pretrained(
            pretrained_model_name_or_path
        )
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path
        ).to(self.whisper_device)
        self.is_initialized = True

    def get_whisper_features(self) -> torch.Tensor:
        """Preprocess audio signal as per the whisper model's training config.

        Returns
        -------
        torch.Tensor
            The prepinput features of the audio signal. Shape: (1, channels, seq_len)
        """
        import torch

        if not self.is_initialized:
            self.setup_whisper()

        signal = self.to(self.device)
        raw_speech = list(
            (
                signal.clone()
                .resample(self.whisper_processor.feature_extractor.sampling_rate)
                .audio_data[:, 0, :]
                .numpy()
            )
        )

        with torch.inference_mode():
            input_features = self.whisper_processor(
                raw_speech,
                sampling_rate=self.whisper_processor.feature_extractor.sampling_rate,
                return_tensors="pt",
            ).input_features

        return input_features

    def get_whisper_transcript(self) -> str:
        """Get the transcript of the audio signal using the whisper model.

        Returns
        -------
        str
            The transcript of the audio signal, including special tokens such as <|startoftranscript|> and <|endoftext|>.
        """

        if not self.is_initialized:
            self.setup_whisper()

        input_features = self.get_whisper_features()

        with torch.inference_mode():
            input_features = input_features.to(self.whisper_device)
            generated_ids = self.whisper_model.generate(inputs=input_features)

        transcription = self.whisper_processor.batch_decode(generated_ids)
        return transcription[0]

    def get_whisper_embeddings(self) -> torch.Tensor:
        """Get the last hidden state embeddings of the audio signal using the whisper model.

        Returns
        -------
        torch.Tensor
            The Whisper embeddings of the audio signal. Shape: (1, seq_len, hidden_size)
        """
        import torch

        if not self.is_initialized:
            self.setup_whisper()

        input_features = self.get_whisper_features()
        encoder = self.whisper_model.get_encoder()

        with torch.inference_mode():
            input_features = input_features.to(self.whisper_device)
            embeddings = encoder(input_features)

        return embeddings.last_hidden_state
