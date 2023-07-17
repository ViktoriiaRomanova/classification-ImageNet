import os
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from processingDataSet import ImageNetDataset


def fit_eval_epoch(model: DDP,
                   loss_func: nn.Module, device: torch.device, data: DataLoader,
                   optim: Optional[torch.optim.Optimizer] = None
                   ) -> Tuple[torch.tensor, torch.tensor]:
    """
        Make train/eval operations per epoch.

        Returns: loss and accuracy.
    """
    avg_loss, accuracy = 0, 0
    for x_batch, y_batch in tqdm(data):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        if optim is not None: optim.zero_grad()

        y_pred = model(x_batch)
        loss = loss_func(y_pred, y_batch)
        if optim is not None:
            loss.backward()
            optim.step()

        # Calculate average train loss and accuracy
        avg_loss += (loss/len(data)).detach().cpu()

        # !it is not final result, to get real accuracy need to divide it into num_batches
        accuracy += torch.sum(torch.argmax(y_pred, 1) == y_batch) / len(y_batch)

        del x_batch, y_batch, y_pred, loss

    accuracy /= len(data)
    return avg_loss, accuracy


def setup(rank: int, world_size: int, isGPU: bool = True) -> None:
    """Setup the process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    # By default torch is using several processes during forward pass.
    # To get a consistent result (in terms: num occupied vCPU == world_size) set that number to 1.
    # It doesn't have a practical reason, just to verify logic.
    if not isGPU: torch.set_num_threads(1)

    # initialize the process group
    # 'nccl' -- for GPU, 'gloo' -- for CPU!
    dist.init_process_group('nccl' if isGPU else 'gloo', rank = rank, world_size = world_size)


def prepare_dataset(data: ImageNetDataset, rank: int,
                    world_size: int, batch_size: int,
                    seed: int) -> DataLoader:
    """
        Split dataset into N parts.

        Returns: DataLoader instance for current part.
    """
    sampler = DistributedSampler(data, num_replicas = world_size,
                                 rank = rank, shuffle = True,
                                 seed = seed, drop_last = True)
    data_loader = DataLoader(data, batch_size = batch_size,
                             shuffle = False, drop_last = True,
                             sampler = sampler)
    return data_loader


def worker(rank: int, model: nn.Module, world_size: int, train_data: List[Tuple[str, int]],
           val_data: List[Tuple[str, int]], batch_size: int,
           seed: int, epochs: int, isGPU: bool = True) -> None:
    """Describe training process which will be implemented for each worker."""
    # Setup process group, for each worker
    setup(rank, world_size, isGPU)

    # prepare data
    train_loader = prepare_dataset(train_data, rank, world_size, batch_size, seed)
    val_loader = prepare_dataset(val_data, rank, world_size, batch_size, seed)

    device = torch.device(rank) if isGPU else torch.device('cpu')
    model.to(device)

    model = DDP(model, device_ids = [rank] if isGPU else None,
                output_device = rank if isGPU else None,
                find_unused_parameters = False)

    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode = 'max', factor = 0.5,
                                                           patience = 3, cooldown = 5)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(epochs):

        train_loader.sampler.set_epoch(epoch)
        val_loader.sampler.set_epoch(epoch)

        model.train()
        train_loss, train_accuracy = fit_eval_epoch(model, loss_func, device, train_loader, optimizer)

        model.eval()
        with torch.no_grad():
            val_loss, val_accuracy = fit_eval_epoch(model, loss_func, device, val_loader)

        # Share metrics
        metrics = torch.tensor([train_loss / world_size, train_accuracy / world_size,
                                val_loss / world_size, val_accuracy / world_size], device = device)
        dist.all_reduce(metrics, op = dist.ReduceOp.SUM)

        # Making sheduler step if validation accuracy is not rising for some time
        scheduler.step(metrics[3])

        if rank == 0:
            print('train_loss: ', metrics[0], 'val_loss: ', metrics[2], 'train_accuracy: ',
                  metrics[1], 'val_accuracy: ', metrics[3], 'epoch: ', epoch + 1, '/', epochs)

    dist.destroy_process_group()
