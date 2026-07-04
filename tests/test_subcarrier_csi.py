import os
import tempfile
import unittest

import numpy as np

from lambda_rf.tools.read_csi import load_csi_npz
from lambda_rf.utils.array_csi import C_M_S
from lambda_rf.utils.subcarrier_csi import (
    build_subcarrier_csi_fields,
    centered_subcarrier_offsets,
    expand_subcarrier_npz,
)


def source_arrays(a_real, a_imag, tau, valid=None):
    a_real_array = np.asarray(a_real, dtype=np.float32)
    return {
        "a_real": a_real_array,
        "a_imag": np.asarray(a_imag, dtype=np.float32),
        "tau": np.asarray(tau, dtype=np.float64),
        "valid": np.ones_like(a_real_array, dtype=bool) if valid is None else np.asarray(valid, dtype=bool),
        "carrier_frequency": np.asarray(C_M_S, dtype=np.float64),
    }


class SubcarrierCsiTest(unittest.TestCase):
    def test_centered_offsets_put_dc_at_floor_half_index(self):
        offsets = centered_subcarrier_offsets(4, 30_000.0)

        np.testing.assert_allclose(offsets, np.array([-60_000.0, -30_000.0, 0.0, 30_000.0]))

    def test_zero_delay_single_link_is_flat_over_subcarriers(self):
        arrays = source_arrays(a_real=[1.0], a_imag=[0.5], tau=[0.0])

        fields = build_subcarrier_csi_fields(arrays, num_subcarriers=4, subcarrier_spacing_hz=1.0)
        h_freq = fields["h_freq_real"] + 1j * fields["h_freq_imag"]

        np.testing.assert_allclose(h_freq, np.full(4, 1.0 + 0.5j), atol=1e-7)

    def test_path_delay_creates_subcarrier_phase_slope(self):
        arrays = source_arrays(a_real=[1.0], a_imag=[0.0], tau=[0.25])

        fields = build_subcarrier_csi_fields(arrays, num_subcarriers=4, subcarrier_spacing_hz=1.0)
        h_freq = fields["h_freq_real"] + 1j * fields["h_freq_imag"]

        expected = np.exp(-1j * 2.0 * np.pi * np.array([-2.0, -1.0, 0.0, 1.0]) * 0.25)
        np.testing.assert_allclose(h_freq, expected, atol=1e-7)

    def test_invalid_paths_do_not_contribute_to_frequency_response(self):
        arrays = source_arrays(
            a_real=[1.0, 100.0],
            a_imag=[0.0, 0.0],
            tau=[0.0, 0.0],
            valid=[True, False],
        )

        fields = build_subcarrier_csi_fields(arrays, num_subcarriers=4, subcarrier_spacing_hz=1.0)
        h_freq = fields["h_freq_real"] + 1j * fields["h_freq_imag"]

        np.testing.assert_allclose(h_freq, np.ones(4), atol=1e-7)

    def test_array_input_sums_paths_per_rx_tx_pair(self):
        arrays = source_arrays(a_real=[0.0], a_imag=[0.0], tau=[0.0])
        arrays["a_mimo_real"] = np.asarray([[[1.0], [2.0]]], dtype=np.float32)
        arrays["a_mimo_imag"] = np.asarray([[[0.5], [-0.25]]], dtype=np.float32)
        arrays["rx_array_shape"] = np.asarray([1, 1], dtype=np.int32)
        arrays["tx_array_shape"] = np.asarray([1, 2], dtype=np.int32)

        fields = build_subcarrier_csi_fields(
            arrays,
            num_subcarriers=3,
            subcarrier_spacing_hz=30_000.0,
            input_mode="array",
        )
        h_freq = fields["h_freq_real"] + 1j * fields["h_freq_imag"]

        self.assertEqual(h_freq.shape, (1, 2, 3))
        np.testing.assert_allclose(h_freq[0, 0], np.full(3, 1.0 + 0.5j), atol=1e-7)
        np.testing.assert_allclose(h_freq[0, 1], np.full(3, 2.0 - 0.25j), atol=1e-7)

    def test_array_input_uses_tau_mimo_when_available(self):
        arrays = source_arrays(a_real=[0.0], a_imag=[0.0], tau=[0.0])
        arrays["a_mimo_real"] = np.asarray([[[1.0], [1.0]]], dtype=np.float32)
        arrays["a_mimo_imag"] = np.asarray([[[0.0], [0.0]]], dtype=np.float32)
        arrays["tau_mimo"] = np.asarray([[[0.0], [0.25]]], dtype=np.float64)

        fields = build_subcarrier_csi_fields(
            arrays,
            num_subcarriers=3,
            subcarrier_spacing_hz=1.0,
            input_mode="array",
        )
        h_freq = fields["h_freq_real"] + 1j * fields["h_freq_imag"]

        np.testing.assert_allclose(h_freq[0, 0], np.ones(3), atol=1e-7)
        expected = np.exp(-1j * 2.0 * np.pi * np.array([-1.0, 0.0, 1.0]) * 0.25)
        np.testing.assert_allclose(h_freq[0, 1], expected, atol=1e-7)
        self.assertEqual(str(fields["frequency_response_model"]), "array_pair_path_delay_phase_from_snapshot")

    def test_expand_subcarrier_npz_preserves_fields_and_read_csi_detects_frequency_csi(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "csi_000000.npz")
            output_path = os.path.join(tmpdir, "subcarrier", "csi_000000.npz")
            arrays = source_arrays(a_real=[1.0], a_imag=[0.0], tau=[0.0])
            np.savez(input_path, **arrays)

            expand_subcarrier_npz(
                input_path=input_path,
                output_path=output_path,
                num_subcarriers=4,
                subcarrier_spacing_hz=30_000.0,
                profile_name="debug_30k_4",
            )

            with np.load(output_path, allow_pickle=False) as data:
                self.assertIn("a_real", data.files)
                self.assertIn("h_freq_real", data.files)
                self.assertEqual(data["h_freq_real"].shape, (4,))
                self.assertEqual(int(data["num_subcarriers"]), 4)

            summary = load_csi_npz(output_path)
            self.assertEqual(summary["freq_shape"], (4,))
            self.assertIn("freq_power", summary)


if __name__ == "__main__":
    unittest.main()
