import logging
import torch.nn.functional as F
from torch.autograd import Variable
from fastprogress import master_bar, progress_bar

from loss import f1_loss
from evaluate import evaluate
import utils


def train_model(model, dataloader, loss_fn, optimizer, scheduler, params):

    # convert model to train, avoid model state is eval
    model.train()

    total = len(dataloader)
    running_loss = 0.0
    running_f1_loss = 0.0
    for j in progress_bar(range(total), parent=params.mb):
        for i, (train_batch, labels_batch) in enumerate(dataloader):
            # move to GPU if available
            if params.cuda:
                train_batch, labels_batch = train_batch.cuda(
                    async=True), labels_batch.cuda(async=True)

            # convert to torch Variables
            train_batch, labels_batch = Variable(
                train_batch), Variable(labels_batch)

            output_batch = model(train_batch)
            loss = loss_fn(output_batch, labels_batch)

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            # adjust learnign rate
            scheduler.step()

            running_loss += loss.item() * train_batch.size(0)
            output_batch = F.sigmoid(output_batch)
            running_f1_loss += f1_loss(output_batch,
                                       labels_batch) * train_batch.size(0)

    train_loss = running_loss / total
    train_f1_loss = running_f1_loss / total

    return train_loss, train_f1_loss


def train_and_evaluate(model, dataloaders, optimizer, loss_fn, scheduler, params):

    best_val_f1_loss = 0.0

    mb = master_bar(range(params.num_epochs))

    params.mb = mb

    for epoch in mb:
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        train_loss, train_f1_loss = train_model(
            model, dataloaders["train"], loss_fn, optimizer, scheduler, params)
        val_loss, val_f1_loss = evaluate(
            model, dataloaders["val"], loss_fn, params)

        list = []
        list.append(train_loss)
        list.append(train_f1_loss)
        list.append(val_loss)
        list.append(val_f1_loss)
        utils.append_train_metrics(list)

        logging.info("Epoch {}/{}. Train loss: {}, Train F1 loss: {}. Val loss: {}, Val F1 loss: {}".format(
            epoch + 1, params.num_epochs, train_loss, train_f1_loss, val_loss, val_f1_loss))

        is_best = val_f1_loss >= best_val_f1_loss

        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint=params.model_dir)

        if is_best:
            logging.info("- Found new best f1 loss")

            best_val_f1_loss = val_f1_loss
