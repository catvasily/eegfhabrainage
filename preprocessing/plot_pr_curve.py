"""
Precision-Recall curve plotting utilities.
"""
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import precision_recall_curve, average_precision_score


def plot_and_save_pr_curve(
    Y,
    y_score,
    step_name='xgboost',
    outfname=None,
    show_plot=True,
    target_label=None,
    physician=None,
):
    """
    Plot and save Precision-Recall curve using seaborn.

    Args:
        Y(ndarray): True binary labels (0, 1)
        y_score(ndarray): Target scores or probabilities for positive class
        step_name(str): Name of the step for output filename (used when outfname is None)
        outfname(str | Path | None): Full pathname for output image
        show_plot(bool): Whether to display the plot
        target_label(str | None): Label name to include in plot title
        physician(str | list | None): Physician name(s) to include in plot title

    Returns:
        outfname(Path): Path to saved figure
    """
    # Compute precision-recall curve
    precision, recall, _ = precision_recall_curve(Y, y_score)
    ap = average_precision_score(Y, y_score)

    # Plot with seaborn styling
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.step(recall, precision, where='post', linewidth=2, label='PR curve')
    ax.fill_between(recall, precision, step='post', alpha=0.2)
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    label_text = f' - {target_label}' if target_label else ''
    physician_text = _physician_label(physician)
    physician_suffix = f' [{physician_text}]' if physician_text else ''
    ax.set_title(f'Precision-Recall Curve{label_text} (AP={ap:.3f}){physician_suffix}', fontsize=14)
    ax.legend(loc='best')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Save figure
    if outfname is None:
        outfname = Path.cwd() / f'pr_curve_{step_name}.png'
    else:
        outfname = Path(outfname)

    outfname.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfname, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    
    plt.close(fig)
    
    return outfname


def _physician_label(value):
    """Return a display string for physician name(s), or empty string if not available."""
    if value is None:
        return ''
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple, set)):
        names = [str(v).strip() for v in value if v is not None and str(v).strip()]
        return ', '.join(names)
    return str(value).strip()
