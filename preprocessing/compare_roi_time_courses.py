"""
A test script that compares ROI time course arrays created by
write_roi_time_courses().
"""

import argparse
import numpy as np

from do_src_reconstr import read_roi_time_courses

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare 'label_tcs' arrays from two ROI .hdf5 files created "
            "by write_roi_time_courses() and report maximal relative difference."
        )
    )
    parser.add_argument("file_a", help="Path to first .hdf5 ROI file")
    parser.add_argument("file_b", help="Path to second .hdf5 ROI file")
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-12,
        help="Minimum denominator for relative difference (default: 1e-12)",
    )

    args = parser.parse_args()

    if args.eps <= 0:
        raise ValueError("--eps must be > 0")

    label_tcs_a = read_roi_time_courses(args.file_a)[0]
    label_tcs_b = read_roi_time_courses(args.file_b)[0]

    if label_tcs_a.shape != label_tcs_b.shape:
        raise ValueError(
            "Shape mismatch for label_tcs arrays: "
            f"{label_tcs_a.shape} vs {label_tcs_b.shape}"
        )

    if label_tcs_a.size == 0:
        print("label_tcs arrays are empty; nothing to compare.")
        return

    mean_rel_diff = _mean_relative_difference(label_tcs_a, label_tcs_b, args.eps)
    std_rel_diff = _std_relative_difference(label_tcs_a, label_tcs_b, args.eps)
    median_rel_diff = _median_relative_difference(label_tcs_a, label_tcs_b, args.eps)
    mad_rel_diff = _mad_relative_difference(label_tcs_a, label_tcs_b, args.eps)
    max_rel_diff, flat_idx = _max_relative_difference(label_tcs_a, label_tcs_b, args.eps)
    idx = np.unravel_index(flat_idx, label_tcs_a.shape)

    a_val = float(label_tcs_a[idx])
    b_val = float(label_tcs_b[idx])
    abs_a_val = abs(a_val)
    abs_b_val = abs(b_val)
    median_abs_a = float(np.median(np.abs(label_tcs_a)))
    median_abs_b = float(np.median(np.abs(label_tcs_b)))
    abs_diff = abs(abs_a_val - abs_b_val)

    print(f"Compared label_tcs arrays with shape {label_tcs_a.shape}")
    print(f"Mean relative difference: {mean_rel_diff:.6e}")
    print(f"Median relative difference: {median_rel_diff:.6e}")
    print(f"MAD relative difference: {mad_rel_diff:.6e}")
    print(f"Std relative difference: {std_rel_diff:.6e}")
    print(f"Max relative difference: {max_rel_diff:.6e}")
    print(f"At index: {idx}")
    print(f"Value in file_a: {a_val:.12e}")
    print(f"Value in file_b: {b_val:.12e}")
    print(f"|Value| in file_a: {abs_a_val:.12e}")
    print(f"|Value| in file_b: {abs_b_val:.12e}")
    print(f"Median |value| in file_a: {median_abs_a:.12e}")
    print(f"Median |value| in file_b: {median_abs_b:.12e}")
    print(f"Absolute difference of |values|: {abs_diff:.12e}")


def _relative_differences(a, b, eps):
    """Return element-wise relative differences array.

    Relative difference is computed as
        abs(abs(a) - abs(b)) / max(sqrt(abs(a) * abs(b)), eps)
    so that near-zero values do not cause division-by-zero errors.
    """
    abs_a = np.abs(a)
    abs_b = np.abs(b)
    denom = np.maximum(np.sqrt(abs_a * abs_b), eps)
    return np.abs(abs_a - abs_b) / denom


def _max_relative_difference(a, b, eps):
    """Return max element-wise relative difference and its flat index."""
    rel = _relative_differences(a, b, eps)

    flat_idx = int(np.argmax(rel))
    return float(rel.flat[flat_idx]), flat_idx


def _mean_relative_difference(a, b, eps):
    """Return mean element-wise relative difference."""
    rel = _relative_differences(a, b, eps)
    return float(np.mean(rel))


def _std_relative_difference(a, b, eps):
    """Return standard deviation of element-wise relative differences."""
    rel = _relative_differences(a, b, eps)
    return float(np.std(rel))


def _median_relative_difference(a, b, eps):
    """Return median of element-wise relative differences."""
    rel = _relative_differences(a, b, eps)
    return float(np.median(rel))


def _mad_relative_difference(a, b, eps):
    """Return median absolute deviation of element-wise relative differences."""
    rel = _relative_differences(a, b, eps)
    median_rel = np.median(rel)
    return float(np.median(np.abs(rel - median_rel)))

if __name__ == "__main__":
    main()
