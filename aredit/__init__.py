"""AR-Edit: streaming video editing on the Reward-Forcing backbone.

This codebase is based on https://github.com/JaydenLyh/Reward-Forcing.

Modules
-------
config            AREditConfig: editing hyper-parameters (defaults = the recipe) and their CLI flags
hook_state        HookState: state shared between the pipeline and the patched attentions
attention_patches patched self-/cross-attention forwards (V store, commit, anchor, text boost, capture)
edit_mask         edit-mask math and the forward hook that builds the anchor mask
prompt_alignment  Needleman-Wunsch alignment of the prompts -> token sets
pipeline          AREditPipeline: source pass -> commit / anchor denoising -> context rerun
runner            AREditRunner: whole-video driver (VAE encode, chunk loop, decode)
video_io          mp4 / prompt reading and result writing
metrics           evaluation metrics: Tonal Fidelity (TF) and TF-u
"""

from aredit.config import AREditConfig
from aredit.prompt_alignment import PromptAlignment, align_prompts

__all__ = ["AREditConfig", "PromptAlignment", "align_prompts", "AREditPipeline", "AREditRunner"]


def __getattr__(name):
    if name == "AREditPipeline":
        from aredit.pipeline import AREditPipeline
        return AREditPipeline
    if name == "AREditRunner":
        from aredit.runner import AREditRunner
        return AREditRunner
    raise AttributeError(name)
