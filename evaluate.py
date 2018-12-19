from torch.autograd import Variable
import torch.nn.functional as F

from loss import f1_loss

def evaluate(model, dataloader, loss_fn, params):
    '''Evaluate the model on `num_steps` batches.
    
    Args:
      model: (torch.nn.Module) the neural network
    '''
    model.eval()

    total = len(dataloader)
    
    running_loss = 0.0
    running_f1_loss = 0.0

    for data_batch, labels_batch in dataloader:
        # move to GPU if available
        if params.cuda:
            data_batch, labels_batch = data_batch.cuda(
                async=True), labels_batch.cuda(async=True)

        # convert to torch Variables
        data_batch, labels_batch = Variable(
            data_batch), Variable(labels_batch)

        output_batch = model(data_batch)
        loss = loss_fn(output_batch, labels_batch)

        running_loss += loss.item() * data_batch.size(0)
        
        output_batch = F.sigmoid(output_batch)
        running_f1_loss +=  f1_loss(output_batch, labels_batch) * data_batch.size(0)

    eval_loss = running_loss / total
    eval_f1_loss = running_f1_loss / total

    return eval_loss, eval_f1_loss


        
