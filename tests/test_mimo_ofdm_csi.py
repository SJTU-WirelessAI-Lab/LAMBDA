import os
import tempfile
import unittest

import numpy as np

from lambda_rf.generate_mimo_ofdm_csi import expand_mimo_ofdm_npz
from lambda_rf.tools.read_csi import load_csi_npz
from lambda_rf.utils.array_csi import C_M_S


class MimoOfdmCsiTest(unittest.TestCase):
    def test_expand_mimo_ofdm_npz_writes_final_mimo_and_frequency_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "csi_000000.npz")
            output_path = os.path.join(tmpdir, "mimo_ofdm", "csi_000000.npz")
            np.savez(
                input_path,
                a_real=np.asarray([1.0], dtype=np.float32),
                a_imag=np.asarray([0.0], dtype=np.float32),
                theta_t=np.asarray([np.pi / 2.0], dtype=np.float32),
                phi_t=np.asarray([0.0], dtype=np.float32),
                theta_r=np.asarray([np.pi / 2.0], dtype=np.float32),
                phi_r=np.asarray([0.0], dtype=np.float32),
                tau=np.asarray([0.0], dtype=np.float64),
                doppler=np.asarray([0.0], dtype=np.float32),
                valid=np.asarray([True], dtype=bool),
                carrier_frequency=np.asarray(C_M_S, dtype=np.float64),
            )

            expand_mimo_ofdm_npz(
                input_path=input_path,
                output_path=output_path,
                tx_shape=(1, 2),
                rx_shape=(1, 1),
                num_subcarriers=4,
                subcarrier_spacing_hz=30_000.0,
                profile_name="debug_30k_4",
            )

            with np.load(output_path, allow_pickle=False) as data:
                self.assertIn("a_mimo_real", data.files)
                self.assertIn("h_freq_real", data.files)
                self.assertEqual(data["a_mimo_real"].shape, (1, 2, 1))
                self.assertEqual(data["h_freq_real"].shape, (1, 2, 4))
                self.assertEqual(str(data["csi_product"]), "mimo_ofdm")

            summary = load_csi_npz(output_path)
            self.assertEqual(summary["mimo_shape"], (1, 2, 1))
            self.assertEqual(summary["freq_shape"], (1, 2, 4))


if __name__ == "__main__":
    unittest.main()
