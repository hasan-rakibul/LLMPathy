import matplotlib.pyplot as plt
import scienceplots

from lightning.pytorch.utilities import rank_zero_only

@rank_zero_only
def log_info(logger, msg):
    logger.info(msg)

@rank_zero_only
def log_debug(logger, msg):
    logger.debug(msg)

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
