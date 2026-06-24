from pathlib import Path
import unittest

from lambda_rf import paths


class PathTagTest(unittest.TestCase):
    def test_detailed_weather_tag_matches_csi_output_tag(self):
        weather = {
            "kind": "snow",
            "snow_rate_mm_h": 2.0,
            "snow_is_lwe": True,
            "snow_model": "gunn_east",
            "snow_type_override": "dry",
            "T_K": 268.15,
        }

        self.assertEqual(
            paths.make_weather_tag(weather),
            "snow_gunn_east_dry_R2p0mmh_lwe_gas_p676",
        )
        self.assertEqual(
            paths.make_csi_output_tag(f_c_hz=60e9, pol="V", weather=weather),
            "f60p0GHz_V_snow_gunn_east_dry_R2p0mmh_lwe_gas_p676",
        )

    def test_clear_keeps_plain_csi_tag(self):
        self.assertEqual(
            paths.make_csi_output_tag(
                f_c_hz=60e9,
                pol="V",
                weather={"kind": "clear", "include_gaseous_absorption": True},
            ),
            "f60p0GHz_V",
        )

    def test_fog_weather_tag_preserves_small_water_density_precision(self):
        self.assertEqual(
            paths.make_weather_tag(
                {
                    "kind": "cloud_fog",
                    "liquid_water_density_g_m3": 0.05,
                    "include_gaseous_absorption": True,
                }
            ),
            "fog_M0p05gpm3_gas_p676",
        )

    def test_single_link_csi_npz_dir_is_frequency_tag_dir(self):
        self.assertFalse(paths.csi_npz_dir().endswith("multi_path_npz"))
        self.assertEqual(Path(paths.csi_npz_dir()).name, paths.make_csi_output_tag())

    def test_data_export_trajectory_root_includes_data_weather_layer(self):
        root = Path(paths.trajectory_root())
        self.assertIn("Sunny", root.parts)

    def test_mimo_ofdm_csi_dir_is_csi_sibling(self):
        output_dir = paths.mimo_ofdm_csi_npz_dir(
            rx_shape=(1, 1),
            tx_shape=(4, 4),
            profile_name="sub6_30k_1024",
        )

        self.assertIn("/mimo_ofdm_csi/", output_dir)
        self.assertIn("/rx1x1_tx4x4/sub6_30k_1024", output_dir)
        self.assertFalse(output_dir.endswith("multi_path_npz"))


if __name__ == "__main__":
    unittest.main()
