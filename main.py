from main_lora import enable_disable_lora
from test import test
from train import train
from data import train_dataloader
from model import heavynetwork, device
import torch


if __name__ == '__main__':
    net = heavynetwork().to(device)
    train(train_dataloader, net, device, epochs=1, total_iterations_limit=100)
    torch.save(net, 'model.pth')
    net = torch.load('model.pth')
    enable_disable_lora(net, None, enabled=True)
    test(net)
    print('Testing finished')

    enable_disable_lora(net, None, enabled=False)
    test(net)
    print('Testing finished')
