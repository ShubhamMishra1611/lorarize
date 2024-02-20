import torch
import json

def save_weights(model, path):
    original_weights = {}
    for name, param in model.named_parameters():
        original_weights[name] = param.data.clone()
    with open(path, 'w') as f:
        json.dump(original_weights, f)

if __name__ == '__main__':
    model = torch.load('model.pth')
    save_weights(model, 'original_weights.json')