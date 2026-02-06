"""Lightning callbacks for LLaVA-Next fine-tuning."""

from lightning.pytorch.callbacks import Callback


class PushToHubCallback(Callback):
    """Push model and processor to Hugging Face Hub after each epoch and at training end."""

    def __init__(self, repo_id: str):
        super().__init__()
        self.repo_id = repo_id

    def on_train_epoch_end(self, trainer, pl_module):
        print(f"Pushing model to the hub, epoch {trainer.current_epoch}")
        pl_module.model.push_to_hub(
            self.repo_id,
            commit_message=f"Training in progress, epoch {trainer.current_epoch}",
        )

    def on_train_end(self, trainer, pl_module):
        print("Pushing model to the hub after training")
        pl_module.processor.push_to_hub(
            self.repo_id, commit_message="Training done"
        )
        pl_module.model.push_to_hub(
            self.repo_id, commit_message="Training done"
        )
