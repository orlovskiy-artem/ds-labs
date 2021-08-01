import keras as ks
import matplotlib.pyplot as plt
import numpy as np
from functools import reduce

def plot_model(model,**kwargs):
    return ks.utils.plot_model(model,show_shapes=True,**kwargs)

def get_shallow_encoded_decoded(X,encoder,decoder,depth = 1):
    """
    :param X: batch of images (list of images)
    :encoder: the encoder
    :decoder: the decoder
    :param depth: number of encoded decoded blocks
    """
    assert depth!=0
    for i in range(1,depth+1):
        X = encoder.layers[i].call(X)
    encoded = X
    for i in range(depth,0,-1):
        X = decoder.layers[-i].call(X)
    decoded = X
    return encoded,decoded

def get_image_autoencoder_analysis(image,encoder,decoder):
    depth2decoded = {}
    depth2encoded = {}
    depth = len(encoder.layers)-1
    for i in range(depth):
        """len(encoder.layers)-1 because one layer is Input layer """
        encoded,decoded = get_shallow_encoded_decoded(np.array([image]),
                                                      encoder,
                                                      decoder,
                                                      depth=i+1)
        # single image extraction
        encoded,decoded = encoded[0],decoded[0] 
        depth2decoded[i] = decoded
        depth2encoded[i] = encoded
#         check_image_reconstruction(image,encoded,decoded)
        plot_maps(encoded,f"Filter Maps for {i}-th depth")
    image_size = reduce(lambda x,y: x*y,image.shape)
    fig,axs = plt.subplots(2,depth+1)
    axs[0][0].imshow(image)
    axs[0][0].set_title(f"Original {image_size}")
    metrics,text_report = get_metrics(image,image)
    axs[1][0].text(0, 0.5,text_report)
    axs[1][0].axis('off')
    image_size = reduce(lambda x,y: x*y,image.shape)
    for i,decoded in depth2decoded.items():
        image_encoded_size = reduce(lambda x,y: x*y,depth2encoded[i].shape)
        axs[0][i+1].imshow(decoded)
        axs[0][i+1].set_title(f"Reconstruction {image_size/image_encoded_size}x")
        metrics,text_report = get_metrics(image,decoded)
        axs[1][i+1].text(0, 0.5, text_report)
        axs[1][i+1].axis('off')
    fig.suptitle(f"Comparison image plot")

    
def check_image_reconstruction(image,image_encoded,image_decoded):
    image_size = reduce(lambda x,y: x*y,image.shape)
    image_encoded_size = reduce(lambda x,y: x*y,image_encoded.shape)
    fig,axs = plt.subplots(1,2)
    axs[0].imshow(image)
    axs[1].imshow(image_decoded)
    axs[0].set_title(f"Original {image_size}")
    axs[1].set_title(f"Reconstruction {image_encoded_size} {image_size/image_encoded_size}x")
    
def get_metrics(y_true,y_pred):
    bin_crossentropy = np.mean(ks.metrics.binary_crossentropy(y_true,y_pred))
    kl_divergence = np.mean(ks.metrics.kl_divergence(y_true,y_pred))
    mse = np.mean(ks.metrics.mse(y_true,y_pred))
    mae = np.mean(ks.metrics.mae(y_true,y_pred))
    n_round = 4
    text_report = f"""
    bin_crossentropy={bin_crossentropy:.{n_round}f}
    kl_divergence={kl_divergence:.{n_round}f}
    mse={mse:.{n_round}f}
    mae={mae:.{n_round}f}
    """
    metrics = {"bin_crossentropy":bin_crossentropy,
               "kl_divergence":kl_divergence,
               "mse":mse,
               "mae":mae}
    return metrics,text_report

def plot_maps(image_encoded,suptitle = ""):
    filter_nums = image_encoded.shape[-1]
    n_rows = filter_nums//4+int(filter_nums%4!=0)
    n_cols = 4
    fig,axs = plt.subplots(n_rows,n_cols)
    if(n_rows>1):
        for i in range(image_encoded.shape[-1]):
            axs[i//4][i%4].imshow(image_encoded[:,:,i])
    else:
        for i in range(image_encoded.shape[-1]):
            axs[i%4].imshow(image_encoded[:,:,i])
    fig.suptitle(suptitle)
    
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)