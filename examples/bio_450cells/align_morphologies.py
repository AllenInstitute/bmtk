"""
Rotates pyramidal cell SWC morphologies so that the apical dendrite main axis
is aligned with the network +y axis (cortical column depth axis).

For the bio_450cells example, Scnn1a, Rorb, and Nr5a1 cells are biophysical
pyramidal cells whose apical dendrites should point along +y so that:
  - rotation_angle_yaxis (applied at sim time) only spins cells azimuthally,
    keeping all apical dendrites pointing upward in the column.
  - Extracellular potential (ECP) calculations see a physiologically correct
    laminar dipole rather than a random-orientation sphere.

Usage:
    python align_morphologies.py
    python align_morphologies.py --morphology-dir /path/to/morphologies
    python align_morphologies.py --no-backup --output-dir rotated/

Original files are backed up as <name>_original.swc unless --no-backup is given.
"""

import argparse
import os
import shutil

import numpy as np
import pandas as pd

# SWC compartment type codes
_SOMA = 1
_AXON = 2
_BASAL = 3
_APICAL = 4

PYRAMIDAL_MODELS = ["Scnn1a", "Rorb", "Nr5a1"]

DEFAULT_MORPHOLOGY_DIR = os.path.join(
    os.path.dirname(__file__), "..", "bio_components", "morphologies"
)
DEFAULT_NODE_TYPES_CSV = os.path.join(
    os.path.dirname(__file__), "network", "internal_node_types.csv"
)


# ---------------------------------------------------------------------------
# SWC I/O
# ---------------------------------------------------------------------------


def read_swc(filepath):
    """Return (header_lines, data_array).

    data_array columns: [id, type, x, y, z, radius, parent_id]
    """
    header, rows = [], []
    with open(filepath) as fh:
        for line in fh:
            stripped = line.rstrip("\n")
            if stripped.lstrip().startswith("#") or stripped.strip() == "":
                header.append(stripped)
            else:
                parts = stripped.split()
                rows.append(
                    [
                        int(parts[0]),
                        int(parts[1]),
                        float(parts[2]),
                        float(parts[3]),
                        float(parts[4]),
                        float(parts[5]),
                        int(parts[6]),
                    ]
                )
    return header, np.array(rows, dtype=float)


def write_swc(filepath, header, data):
    with open(filepath, "w") as fh:
        for h in header:
            fh.write(h + "\n")
        for row in data:
            fh.write(
                f"{int(row[0])} {int(row[1])} "
                f"{row[2]:.6f} {row[3]:.6f} {row[4]:.6f} "
                f"{row[5]:.6f} {int(row[6])}\n"
            )


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def apical_principal_axis(data):
    """Return unit vector along the apical dendrite main axis, pointing away from soma.

    Uses PCA on all apical (type-4) compartment positions relative to the soma
    centroid.  The sign is chosen so the vector points in the same general
    direction as the apical centroid (i.e. away from the soma).
    """
    soma_pts = data[data[:, 1] == _SOMA, 2:5]
    apical_pts = data[data[:, 1] == _APICAL, 2:5]

    if len(apical_pts) == 0:
        raise ValueError("No apical dendrite compartments (type 4) found in SWC.")

    soma_center = soma_pts.mean(axis=0)
    centered = apical_pts - soma_center

    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    axis = eigenvectors[:, np.argmax(eigenvalues)]  # largest eigenvalue → main axis

    # Orient toward the apical centroid (away from soma)
    if np.dot(axis, centered.mean(axis=0)) < 0:
        axis = -axis

    return axis / np.linalg.norm(axis)


def rotation_matrix_to_y(v):
    """Return 3×3 rotation matrix R such that R @ v ∥ [0, 1, 0].

    Uses Rodrigues' rotation formula.
    """
    target = np.array([0.0, 1.0, 0.0])
    v = np.asarray(v, dtype=float) / np.linalg.norm(v)

    if np.allclose(v, target, atol=1e-9):
        return np.eye(3)

    if np.allclose(v, -target, atol=1e-9):
        # 180° rotation around any perpendicular axis
        perp = (
            np.array([1.0, 0.0, 0.0]) if abs(v[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
        )
        perp -= np.dot(perp, v) * v
        perp /= np.linalg.norm(perp)
        return 2.0 * np.outer(perp, perp) - np.eye(3)

    axis = np.cross(v, target)
    axis /= np.linalg.norm(axis)
    angle = np.arccos(np.clip(np.dot(v, target), -1.0, 1.0))

    K = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def rotate_swc(data, R, pivot):
    """Apply rotation R around pivot to all (x, y, z) coordinates in data."""
    result = data.copy()
    xyz = data[:, 2:5] - pivot
    result[:, 2:5] = (R @ xyz.T).T + pivot
    return result


# ---------------------------------------------------------------------------
# Per-morphology processing
# ---------------------------------------------------------------------------


def align_morphology(swc_path, output_path, backup=True):
    """Rotate morphology so apical dendrites point along +y.  Returns angle (deg) before rotation."""
    if backup and swc_path == output_path:
        backup_path = swc_path.replace(".swc", "_original.swc")
        if not os.path.exists(backup_path):
            shutil.copy2(swc_path, backup_path)
            print(f"  Backup saved → {os.path.basename(backup_path)}")
        else:
            print(f"  Backup already exists, skipping copy.")

    header, data = read_swc(swc_path)

    soma_center = data[data[:, 1] == _SOMA, 2:5].mean(axis=0)
    axis_before = apical_principal_axis(data)
    y_hat = np.array([0.0, 1.0, 0.0])
    angle_before = np.degrees(np.arccos(np.clip(np.dot(axis_before, y_hat), -1.0, 1.0)))

    R = rotation_matrix_to_y(axis_before)
    data_rotated = rotate_swc(data, R, pivot=soma_center)

    axis_after = apical_principal_axis(data_rotated)
    angle_after = np.degrees(np.arccos(np.clip(np.dot(axis_after, y_hat), -1.0, 1.0)))

    print(
        f"  Apical axis before: [{axis_before[0]:+.3f}, {axis_before[1]:+.3f}, {axis_before[2]:+.3f}]"
        f"  →  {angle_before:.1f}° from +y"
    )
    print(
        f"  Apical axis after:  [{axis_after[0]:+.3f}, {axis_after[1]:+.3f}, {axis_after[2]:+.3f}]"
        f"  →  {angle_after:.2f}° from +y"
    )

    write_swc(output_path, header, data_rotated)
    print(f"  Written → {output_path}")
    return angle_before


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--node-types-csv",
        default=DEFAULT_NODE_TYPES_CSV,
        help="Path to internal_node_types.csv",
    )
    parser.add_argument(
        "--morphology-dir",
        default=DEFAULT_MORPHOLOGY_DIR,
        help="Directory containing SWC morphology files",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Write rotated SWC files here instead of overwriting originals",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating _original.swc backup when overwriting in place",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=PYRAMIDAL_MODELS,
        help="Model names to process (default: Scnn1a Rorb Nr5a1)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    morph_dir = os.path.realpath(args.morphology_dir)
    node_types = pd.read_csv(args.node_types_csv, sep=r"\s+")

    targets = node_types[node_types["model_name"].isin(args.models)]
    if targets.empty:
        raise SystemExit(
            f"No matching models found in {args.node_types_csv}.\n"
            f'Available: {list(node_types["model_name"])}'
        )

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    print(f"Morphology dir : {morph_dir}")
    print(f"Models         : {args.models}\n")

    for _, row in targets.iterrows():
        morph_file = row["morphology"]
        if morph_file in ("NULL", "none", "") or not isinstance(morph_file, str):
            print(f'[{row["model_name"]}] no morphology file, skipping.')
            continue

        swc_in = os.path.join(morph_dir, morph_file)
        if not os.path.exists(swc_in):
            print(f'[{row["model_name"]}] file not found: {swc_in}, skipping.')
            continue

        if args.output_dir:
            swc_out = os.path.join(args.output_dir, morph_file)
        else:
            swc_out = swc_in  # overwrite in place

        print(f'[{row["model_name"]}]  {morph_file}')
        align_morphology(swc_in, swc_out, backup=not args.no_backup)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
