from transformers import PretrainedConfig


class AudioVAEconfig(PretrainedConfig):
    def __init__(
        self,
        sample_rate: int=16000,
        enc_kwargs: dict = None,
        semantic_module_kwargs: dict = None,
        dec_kwargs: dict = None,
        hifi_gan_disc_kwargs: dict = None,
        spec_disc_kwargs: dict = None,
        lambda_disc=1.0,
        lambda_mel_loss=15,
        lambda_adv=1.0,
        lambda_feat_match_loss=1.0,
        lambda_semantic=5.0,
        init_method='normal',
        patch_size=-1,
        **kwargs
    ):
        self.sample_rate = sample_rate
        self.enc_kwargs = enc_kwargs
        self.semantic_module_kwargs = semantic_module_kwargs
        self.dec_kwargs = dec_kwargs
        self.hifi_gan_disc_kwargs = hifi_gan_disc_kwargs
        self.spec_disc_kwargs = spec_disc_kwargs
        self.lambda_disc = lambda_disc
        self.lambda_mel_loss = lambda_mel_loss
        self.lambda_adv = lambda_adv
        self.lambda_feat_match_loss = lambda_feat_match_loss
        self.lambda_semantic = lambda_semantic
        self.init_method = init_method
        self.patch_size = patch_size
        super().__init__(**kwargs)
