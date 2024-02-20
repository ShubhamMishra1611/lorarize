from tqdm import tqdm
import torch
import torch.nn as nn
from data import train_dataloader


def train(train_dataloader, net, device, epochs=5, total_iterations_limit=None):
    cross_el = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    total_iterations = 0

    for epoch in range(epochs):
        net.train()
        loss_sum = 0
        num_iterations = 0

        data_iterator = tqdm(train_dataloader, desc=f'Epoch {epoch+1}')

        if total_iterations_limit is not None:
            data_iterator.total = total_iterations_limit

        for data in data_iterator:
            num_iterations += 1
            total_iterations += 1
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            output = net(x.view(-1, 28*28))
            loss = cross_el(output, y)
            loss_sum += loss.item()
            avg_loss = loss_sum / num_iterations
            data_iterator.set_postfix(loss=avg_loss)
            loss.backward()
            optimizer.step()

            if total_iterations_limit is not None and total_iterations >= total_iterations_limit:
                return


if __name__ == '__main__':
    from model import heavynetwork, device
    net = heavynetwork().to(device)
    train(train_dataloader, net, device, total_iterations_limit=1000)
    torch.save(net, 'model.pth')
    print('Training finished')
