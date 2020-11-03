def train_one_epoch():
    for x, label in tqdm(train_loader, desc="Training one Epoch", leave=False):  # iterate over batch
        # set all gradients to zero
        optimizer.zero_grad()
        
        # move all tensors to GPU if available
        x = x.to(device)
        label = label.to(device)
        
        
        # compute prediction for each sample in the batch, compute loss to minimize
        y = lenet(x)
        loss = criterion(y, label)
        
        # backpropagate gradients
        loss.backward()
        
        # take gradient step
        optimizer.step()
        