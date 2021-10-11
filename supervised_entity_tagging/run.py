import numpy as np
import os
import torch
from tqdm import tqdm

from utils.dataloader import construct_dataloaders, _reconstruct_input_labels
from utils.utils import create_optimizer_and_scheduler
from utils.options import parse_arguments

from models.net import EntityTagger


def main():
    opts = parse_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{opts.gpu}"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)
    if opts.gpu.count(",") > 0:
        opts.batch_size = opts.batch_size * (opts.gpu.count(",")+1)
        opts.eval_batch_size = opts.eval_batch_size * (opts.gpu.count(",")+1)
    loaders = construct_dataloaders(opts.root, opts.model_name, opts.batch_size, opts.num_workers, opts.seed)

    model = EntityTagger(
        nclass=len(loaders["train"].dataset.label2id),
        model_name=opts.model_name
    )

    if opts.gpu.count(",") > 0:
        model = torch.nn.DataParallel(model)

    device = torch.device('cuda:0') if torch.cuda.is_available() and (not opts.no_gpu) else torch.device('cpu')
    model.to(device)

    if not opts.test_only: # you need to add code to load model if you only want to run test
        optimizer, scheduler = create_optimizer_and_scheduler(model, opts.learning_rate, opts.decay, opts.warmup_step, len(loaders["train"]) * opts.train_epoch)

        # this is just training on a fixed number of epochs. You need to implement yourself if you want to select the best checkpoints according to the dev set performance.
        for epoch in range(opts.train_epoch):
            iterator = tqdm(loaders["train"])
            epoch_loss = 0.
            for idx, (encodings, labels) in enumerate(iterator):
                try:
                    encodings = encodings.to(device)
                except Exception as e:
                    encodings = {key: val.to(device) for key, val in encodings.items()}
                labels = labels.to(device)
                outputs = model(encodings, labels)
                optimizer.zero_grad()
                outputs["loss"].backward()
                optimizer.step()
                scheduler.step()
                epoch_loss += outputs["loss"].item()
                iterator.set_postfix({"loss": epoch_loss / (idx + 1)})

    with torch.no_grad():
        iterator = tqdm(loaders["test"])
        epoch_loss = 0.
        predictions = []
        test_dataset = loaders["test"].dataset
        for idx, (encodings, labels) in enumerate(iterator):
            try:
                inputs = encodings.to(device)
            except Exception as e:
                inputs = {key: val.to(device) for key, val in encodings.items()}
            labels = labels.to(device)
            outputs = model(inputs, labels)
            encodings = test_dataset.collate_fn(test_dataset.data[idx * opts.batch_size: (idx+1) * opts.batch_size])[0]
            prediction_labels = _reconstruct_input_labels(encodings, outputs["prediction"], loaders["test"].dataset.id2label)
            predictions.extend(prediction_labels)
            epoch_loss += outputs["loss"].item()
            iterator.set_postfix({"loss": epoch_loss / (idx + 1)})
    outputs = loaders["test"].dataset.dumps_outputs(predictions)
    with open(os.path.join(opts.log_dir, "test_output.txt"), "wt") as fp:
        fp.write(outputs)
    torch.save(model.state_dict(), os.path.join(opts.log_dir, "model.ckpt"))


if __name__ == "__main__":
    main()