import torch


def save_checkpoint(ckpt_path, model, optimizer):
    dct = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict()
    }
    print('Saving checkpoint to {}'.format(ckpt_path))
    torch.save(dct, ckpt_path)


def load_checkpoint(ckpt_path, model, optimizer=None):
    dct = torch.load(ckpt_path)
    model.load_state_dict(dct['model'])
    if optimizer:
        optimizer.load_state_dict(dct['optim'])