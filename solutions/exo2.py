def evaluate(dataloader):
    # define accumulators
    total_loss = 0
    total_accuracy = 0
    num_samples = 0

    with torch.no_grad():  # no need to compute gradients during evaluation, this will be faster
        for x, label in tqdm(dataloader, desc="Evaluating", leave=False):
            # move to GPU if available
            x = x.to(device)
            label = label.to(device)
        
            # compute network output
            y = lenet(x)
            
            # compute loss
            loss = criterion(y, label)
            
            # compute accuracy
            pred = torch.argmax(y, dim=1)
            accuracy = torch.sum(pred == label)
            
            # update accumulators
            num_samples += len(label)  # number of samples in the batch
            total_loss += loss.detach().cpu().numpy() * len(label)  # by default, pytorch averages loss over batch
            total_accuracy += accuracy.detach().cpu().numpy()

            
    return total_loss / num_samples, total_accuracy / num_samples
