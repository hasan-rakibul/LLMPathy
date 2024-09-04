import os
import matplotlib.pyplot as plt
import scienceplots

import torch
import numpy as np
import random

def plot(x, y, y2=None, xlabel=None, ylabel=None, legend=[], save=False, filename=None):
    """Plot data points"""
    plt.style.use(['science'])
    fig, ax = plt.subplots(1, 1)
    
    ax.plot(x, y)
    if y2 is not None:
        ax.plot(x, y2)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(legend)
    
    if save:
        plt.savefig(fname=filename+'.pdf', format='pdf', bbox_inches='tight')
        print(f"Saved as {filename}.pdf")
        
    fig.show()

def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def resolve_checkpoint(checkpoint_dir):
    items = os.listdir(checkpoint_dir)
    checkpoint = [item for item in items if item.startswith("checkpoint")]
    assert len(checkpoint) == 1, "checkpoint_dir should contain only one checkpoint"
    return os.path.join(checkpoint_dir, checkpoint[0])