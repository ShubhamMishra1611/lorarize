import argparse
from main_lora import enable_disable_lora, main_param
from test import test
from train import train
from data import train_dataloader
from model import heavynetwork, device
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='Path to the model file')
    args = parser.parse_args()

    net = torch.load(args.model_path)
    net = main_param(net)
    enable_disable_lora(net, None, enabled=True)
    test(net)
    print('Testing finished')
