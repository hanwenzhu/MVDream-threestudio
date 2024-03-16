import json
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModel

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessor, hash_prompt
from threestudio.utils.misc import cleanup
from threestudio.utils.textual_inversion import load_textual_inversion, maybe_convert_prompt
from threestudio.utils.typing import *


@threestudio.register("stable-diffusion-prompt-processor")
class StableDiffusionPromptProcessor(PromptProcessor):
    @dataclass
    class Config(PromptProcessor.Config):
        # paths to learned embeddings for textual inversion
        pretrained_model_name_or_path_textual_inversion: Optional[List[str]] = None

    cfg: Config

    ### these functions are unused, kept for debugging ###
    def configure_text_encoder(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.pretrained_model_name_or_path, subfolder="tokenizer", use_fast=False
        )
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.cfg.pretrained_model_name_or_path, subfolder="text_encoder"
        ).to(self.device)

        for p in self.text_encoder.parameters():
            p.requires_grad_(False)
        
        # load textual inversion
        if self.cfg.pretrained_model_name_or_path_textual_inversion is not None:
            load_textual_inversion(
                self.cfg.pretrained_model_name_or_path_textual_inversion,
                self.tokenizer,
                self.text_encoder
            )

    def destroy_text_encoder(self) -> None:
        del self.tokenizer
        del self.text_encoder
        cleanup()

    def get_text_embeddings(
        self, prompt: Union[str, List[str]], negative_prompt: Union[str, List[str]]
    ) -> Tuple[Float[Tensor, "B 77 768"], Float[Tensor, "B 77 768"]]:
        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]
        # Tokenize text and get embeddings
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        uncond_tokens = self.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )

        with torch.no_grad():
            text_embeddings = self.text_encoder(tokens.input_ids.to(self.device))[0]
            uncond_text_embeddings = self.text_encoder(
                uncond_tokens.input_ids.to(self.device)
            )[0]

        return text_embeddings, uncond_text_embeddings

    ###

    @staticmethod
    def spawn_func(cfg, prompts, cache_dir):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.pretrained_model_name_or_path, subfolder="tokenizer", use_fast=False
        )
        text_encoder = CLIPTextModel.from_pretrained(
            cfg.pretrained_model_name_or_path,
            subfolder="text_encoder",
            device_map="auto",
        )
        
        # load textual inversion
        if cfg.pretrained_model_name_or_path_textual_inversion is not None:
            load_textual_inversion(
                cfg.pretrained_model_name_or_path_textual_inversion,
                tokenizer,
                text_encoder
            )
            # Convert prompt for multi-vector
            prompts = maybe_convert_prompt(prompts)

        with torch.no_grad():
            tokens = tokenizer(
                prompts,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            )
            text_embeddings = text_encoder(tokens.input_ids.to(text_encoder.device))[0]

        for prompt, embedding in zip(prompts, text_embeddings):
            torch.save(
                embedding,
                os.path.join(
                    cache_dir,
                    f"{hash_prompt(cfg.pretrained_model_name_or_path, prompt)}.pt",
                ),
            )

        del text_encoder
