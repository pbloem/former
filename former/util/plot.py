import torch

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def cosine_heatmap(vectors, ax=plt.gca(), cm='inferno'):
    """
    Plot a heatmap of the cosine similarities between the given vectors
    :param vectors:
    :param axes: Default is current axes
    :return:
    """

    num, dim = vectors.size()

    norms = vectors.norm(dim=1, keepdim=True)

    dots = torch.mm(vectors, vectors.t())
    div  = torch.mm(norms, norms.t())

    im = (dots/div).cpu().numpy()

    ax.imshow(im)

    return

