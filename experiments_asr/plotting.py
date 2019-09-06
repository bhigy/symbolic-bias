# matplotlib related
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def plot_attention(att, labels, ali=None, plot_scale=False, fontsize_labels=9,
                   fontsize_titles=10, pad=-10, labels_ds_rate=4.):
    if ali is not None:
        fig, axes = plt.subplots(2, 1)
        main_axis = axes[1]
    else:
        fig, axes = plt.subplots(1, 1)
        main_axis = axes
    # Add alignments
    if ali is not None:
        empty = np.zeros([1, att.shape[1]])
        axes[0].imshow(empty, cmap='Blues', vmin=0, vmax=1)
        starts = []
        phonemes = []
        for i, p in enumerate(ali):
            f = p['first'] / labels_ds_rate
            l = p['last'] / labels_ds_rate
            xline = l - 0.5
            if i < len(ali) - 1:
                axes[0].axvline(xline, color='red')
            else:
                l = math.ceil(l)
            starts.append((f + l) / 2 - 0.5)
            phonemes.append(p['phoneme'])
        axes[0].set_xticks(starts)
        axes[0].set_xticklabels(phonemes)
        # switch y ticks off
        axes[0].tick_params(axis='both', which='both', bottom=False, left=False, labelleft=False, pad=pad,
                            labelsize=fontsize_labels)
        axes[0].set_anchor('SE')
    # Main plot
    cax = main_axis.imshow(att, cmap='Blues', interpolation='nearest', vmin=0, vmax=1)
    main_axis.set_facecolor('black')
    # set y-axis labels
    main_axis.set_yticks(range(len(labels)))
    main_axis.set_yticklabels(labels)
    main_axis.tick_params(axis='both', which='both', labelsize=fontsize_labels)
    main_axis.set_xlabel('Input frames', labelpad=10, fontsize=fontsize_titles)
    h = main_axis.set_ylabel('Output\ntokens', labelpad=10, fontsize=fontsize_titles)
    main_axis.set_anchor('NE')
    if plot_scale:
        cb = fig.colorbar(cax)
        cb.ax.tick_params(axis='both', which='both', labelsize=fontsize_labels)
    else:
        fig.subplots_adjust(hspace=0.05)
    return fig, axes
