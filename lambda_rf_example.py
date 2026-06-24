"""Minimal public example for LAMBDA CSI utilities."""

from __future__ import annotations

from lambda_rf.cli import list_scenarios


class _Args:
    config = None


def main() -> None:
    print("Configured utility scenarios:")
    list_scenarios(_Args())
    print()
    print("Example commands:")
    print("  python -m lambda_rf read-csi path/to/csi_000000.npz")
    print("  python -m lambda_rf mimo-ofdm-csi --input-dir path/to/csi/f60p0GHz_V --tx-shape 4,4 --rx-shape 1,1 --profile sub6_30k_1024")
    print("  python -m lambda_rf radar --input-dir path/to/csi/f60p0GHz_V --imu-dir path/to/imu")
    print("  python -m lambda_rf radar-vis --input-dir path/to/radar_raw/f60p0GHz_V")


if __name__ == "__main__":
    main()
