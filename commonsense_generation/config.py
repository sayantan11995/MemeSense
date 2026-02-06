"""Configuration for LLaVA-Next fine-tuning."""

from dataclasses import dataclass, field


PROMPT_TEMPLATE = """\
As an AI assistant for social media content moderation, your role is to analyze and prevent \
the posting of harmful, offensive, hateful, vulgar, misogynistic, or otherwise inappropriate \
content on public platforms.

Task: Carefully analyze and interpret the given image, paying close attention to both visual \
elements and any text present.

1. Meme Description:
Provide a detailed description of the meme, including any text, imagery, or symbolic references.

2. Commonsense Analysis:
* Assess whether the meme has the potential to be perceived as vulgar, harmful, or unethical.
* Identify and categorize up to 5 specific commonsense parameters from the predefined list below.
* Each selected category should be explicitly justified with a brief explanation.

Predefined Commonsense Categories (Select Up to 5):
* Vulgarity: Use of obscene language, inappropriate tone, or implied innuendos.
* Violence: Depictions of harm, coercion, or references to physical aggression.
* Sexual Content: Explicit or suggestive imagery, innuendos, or nudity.
* Hate Speech: Insulting, harassing, or offensive content targeting individuals or groups.
* Body Shaming: Objectification, fat-shaming, or insensitive references to disabilities.
* Cultural Sensitivity: Stereotypes, appropriation, or misrepresentation of cultural elements.
* Mental Health Impact: Content that manipulates emotions, triggers trauma, or exploits grief.
* Humor Appropriateness: Use of humor in a way that mocks, belittles, or distorts serious topics.
* Misinformation: Spreading misleading narratives, false claims, or deceptive dialogue.
* Substance Abuse: References to drugs, alcohol, or promoting substance use.
* Child Exploitation: Content that inappropriately depicts or exploits children.
* Public Decorum & Privacy: Violations of privacy, public sensitivity, or personal reputation concerns.
* Stereotyping: Generalizations that reinforce racial, gender, or societal biases.
* Misogyny: Content promoting gender-based discrimination, sexism, or demeaning women.
* Religious Sensitivity: Content that disrespects religious beliefs, symbols, or historical context.

3. Intervention Recommendation:
* If the meme is deemed inappropriate, justify why it should not be posted publicly.
* If the content is safe, confirm its appropriateness.

Response Format:

Meme Description:
<Provide a detailed description of the meme, including text and images.>

Commonsense Analysis:
- **[Category Name]**: [Justification]
- **[Category Name]**: [Justification]
- **[Category Name]**: [Justification]

Intervention Recommendation:
<Explain whether the meme should be restricted and why.>
"""


@dataclass
class ModelConfig:
    """Model and LoRA configuration."""
    model_id: str = "llava-hf/llava-v1.6-mistral-7b-hf"
    use_lora: bool = False
    use_qlora: bool = True
    lora_r: int = 4
    lora_alpha: int = 8
    lora_dropout: float = 0.1
    max_length: int = 256


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    max_epochs: int = 10
    check_val_every_n_epoch: int = 1
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 8
    lr: float = 1e-4
    batch_size: int = 1
    num_nodes: int = 1
    warmup_steps: int = 5
    result_path: str = "./result"
    verbose: bool = True
    precision: str = "16-mixed"
    limit_val_batches: int = 2
    devices: list = field(default_factory=lambda: [0])


@dataclass
class DataConfig:
    """Dataset configuration."""
    data_dir: str = "../commonsense_labelled_data_new/"
    num_workers: int = 4


@dataclass
class HubConfig:
    """Hugging Face Hub and W&B configuration."""
    repo_id: str = "YOUR-HUB-REPO-TO-PUSH"
    wandb_project: str = "LLaVaNeXT"
    wandb_name: str = "llava-next-demo-cord"
    save_path: str = "saved_model_detailed_prompt"
