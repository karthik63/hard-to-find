import torch
from transformers.optimization import AdamW, get_scheduler
from transformers.trainer_pt_utils import get_parameter_names

def create_optimizer_and_scheduler(model:torch.nn.Module, learning_rate:float, weight_decay:float, warmup_step:int, train_step:int, adam_beta1:float=0.9, adam_beta2:float=0.999, adam_epsilon:float=1e-8):
    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]
    optimizer_kwargs = {
        "lr": learning_rate,
        "betas": (adam_beta1, adam_beta2),
        "eps": adam_epsilon,
    }
    optimizer = AdamW(optimizer_grouped_parameters, **optimizer_kwargs)
    scheduler = get_scheduler(
                "linear",
                optimizer,
                num_warmup_steps=warmup_step,
                num_training_steps=train_step,
            )
    return optimizer, scheduler
