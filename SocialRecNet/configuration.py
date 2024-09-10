"""SocialRecNet config"""

from transformers import PretrainedConfig, LlamaConfig
from transformers import logging

logger = logging.get_logger(__name__)

class SocialRecNetConfig(PretrainedConfig):
    def __init__(
        self, 
        llama_config=None,
        conv_kernel_sizes="5,5,5",
        adapter_inner_dim=512,
        **kwargs
    ):
        super().__init__(**kwargs)

        if llama_config is None:
            llama_config = {}
            logger.info("llama config is None. Initializing the LlamaConfig with default values")
    
        self.llama_config = LlamaConfig(**llama_config).to_dict()

        self.conv_kernel_sizes = conv_kernel_sizes
        self.adapter_inner_dim = adapter_inner_dim