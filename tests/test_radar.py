import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from lambda_rf import config
from lambda_rf.utils.radar import (
    C_M_S,
    ConstantRCSModel,
    H5RCSModel,
    RadarSystem,
    load_csi_paths,
    load_radar_npz,
    radar_one_way_tau_by_antenna,
    resolve_rcs_model_path,
    synthesize_radar_cube,
    virtual_array_positions,
)
from lambda_rf.visualize_radar import compute_radar_maps

RCS_28_PATH = Path(config.SERVER_ROOT) / "assets" / "default_drone_rcs_28ghz.h5"
RCS_60_PATH = Path(config.SERVER_ROOT) / "assets" / "default_drone_rcs_60ghz.h5"
RCS_77_PATH = Path(config.SERVER_ROOT) / "assets" / "default_drone_rcs_77ghz.h5"


class RadarUtilityTest(unittest.TestCase):
    def test_load_radar_npz_preserves_mount_orientation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "radar_000000.npz"
            mount = np.asarray([12.0, 23.0, 34.0])
            np.savez(
                path,
                radar_data=np.zeros((1, 2, 4), dtype=np.complex64),
                radar_params=np.asarray([28.0e9, 1.0, 1.0, 1.0, 4.0, 1.0]),
                radar_mount_yaw_pitch_roll_deg=mount,
            )

            _, info, _ = load_radar_npz(path)

        np.testing.assert_allclose(info["radar_mount_yaw_pitch_roll_deg"], mount)

    def test_default_rcs_asset_is_bundled(self):
        self.assertTrue(RCS_28_PATH.is_file())
        self.assertTrue(RCS_60_PATH.is_file())
        self.assertTrue(RCS_77_PATH.is_file())

    @unittest.skipIf(importlib.util.find_spec("h5py") is None, "h5py is not installed")
    def test_bundled_h5_rcs_can_be_sampled(self):
        model = H5RCSModel(RCS_28_PATH, expected_frequency_hz=28.0e9)
        field = model.get_scattering_amplitude(theta_deg=0.0, phi_deg=0.0)
        self.assertAlmostEqual(field.real, 1.42671406, places=7)
        self.assertAlmostEqual(field.imag, 1.82571649, places=7)
        self.assertAlmostEqual(model.get_rcs(0.0, 0.0), 67.4657488, places=5)
        self.assertEqual(model.incident_polarization, "theta")
        self.assertEqual(model.component, "theta")

    @unittest.skipIf(importlib.util.find_spec("h5py") is None, "h5py is not installed")
    def test_rcs_frequency_mismatch_and_unsupported_band_fail(self):
        model_60 = H5RCSModel(RCS_60_PATH, expected_frequency_hz=60.0e9)
        self.assertTrue(np.isfinite(model_60.get_rcs(0.0, 0.0)))
        self.assertEqual(resolve_rcs_model_path(60.0e9, Path(config.SERVER_ROOT) / "assets"), RCS_60_PATH)
        model_77 = H5RCSModel(RCS_77_PATH, expected_frequency_hz=77.0e9)
        self.assertTrue(np.isfinite(model_77.get_rcs(0.0, 0.0)))
        with self.assertRaisesRegex(ValueError, "does not match"):
            H5RCSModel(RCS_28_PATH, expected_frequency_hz=77.0e9)
        with self.assertRaisesRegex(ValueError, "No calibrated drone RCS model"):
            resolve_rcs_model_path(5.9e9, Path(config.SERVER_ROOT) / "assets")

    def test_radar_equation_amplitude_normalization(self):
        frequency = 28.0e9
        distance_m = 60.0
        sigma_m2 = 4.0
        wavelength = C_M_S / frequency
        one_way_a = wavelength / (4.0 * np.pi * distance_m)
        radar = RadarSystem(
            f_c=frequency,
            bandwidth=1.0e6,
            sample_rate=2.0e6,
            chirp_duration=4.0e-6,
            num_chirps=2,
            tx_power_dbm=30.0,
            tx_gain_db=0.0,
            rx_gain_db=0.0,
        )
        csi_data = {
            "a": np.asarray([one_way_a + 0.0j]),
            "tau": np.asarray([distance_m / C_M_S]),
            "doppler": np.asarray([0.0]),
            "theta_r": np.asarray([np.pi / 2.0]),
            "phi_r": np.asarray([0.0]),
            "theta_t": np.asarray([np.pi / 2.0]),
            "phi_t": np.asarray([0.0]),
        }
        cube = synthesize_radar_cube(
            csi_data=csi_data,
            uav_rotation_l2w=np.eye(3),
            rcs_model=ConstantRCSModel(sigma_m2),
            radar_system=radar,
            antenna_positions_m=np.zeros((1, 3)),
        )
        time_signal = np.fft.ifft(cube, axis=-1)
        expected_power_w = wavelength**2 * sigma_m2 / ((4.0 * np.pi) ** 3 * distance_m**4)
        self.assertAlmostEqual(abs(time_signal[0, 0, 0]), np.sqrt(expected_power_w), delta=1e-12)

    def test_mimo_expanded_csi_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "csi_000000.npz"
            shape = (2, 1, 3)
            np.savez(
                path,
                a_real=np.ones(shape),
                a_imag=np.zeros(shape),
                tau=np.ones(shape),
                theta_r=np.ones(shape),
                phi_r=np.ones(shape),
                theta_t=np.ones(shape),
                phi_t=np.ones(shape),
                carrier_frequency=np.asarray(28.0e9),
            )
            with self.assertRaisesRegex(ValueError, "representative 1x1"):
                load_csi_paths(path)

    def test_synthesize_radar_cube_shape(self):
        radar = RadarSystem(
            f_c=C_M_S,
            bandwidth=4.0,
            sample_rate=8.0,
            chirp_duration=1.0,
            num_chirps=4,
        )
        csi_data = {
            "a": np.asarray([1.0 + 0.0j]),
            "tau": np.asarray([0.0]),
            "doppler": np.asarray([0.0]),
            "theta_r": np.asarray([np.pi / 2.0]),
            "phi_r": np.asarray([0.0]),
            "theta_t": np.asarray([np.pi / 2.0]),
            "phi_t": np.asarray([0.0]),
        }
        ant_pos = virtual_array_positions(radar.f_c, shape=(1, 2), spacing_wavelengths=0.5)

        cube = synthesize_radar_cube(
            csi_data=csi_data,
            uav_rotation_l2w=np.eye(3),
            rcs_model=ConstantRCSModel(1.0),
            radar_system=radar,
            antenna_positions_m=ant_pos,
        )

        self.assertEqual(cube.shape, (2, 4, 8))
        self.assertEqual(cube.dtype, np.complex64)

    def test_chirp_interval_controls_slow_time_phase(self):
        radar = RadarSystem(
            f_c=C_M_S,
            bandwidth=4.0,
            sample_rate=8.0,
            chirp_duration=1.0,
            chirp_interval=1.25,
            num_chirps=4,
        )
        csi_data = {
            "a": np.asarray([1.0 + 0.0j]),
            "tau": np.asarray([0.0]),
            "doppler": np.asarray([0.05]),
            "theta_r": np.asarray([np.pi / 2.0]),
            "phi_r": np.asarray([0.0]),
            "theta_t": np.asarray([np.pi / 2.0]),
            "phi_t": np.asarray([0.0]),
        }
        ant_pos = virtual_array_positions(radar.f_c, shape=(1, 1), spacing_wavelengths=0.5)

        cube = synthesize_radar_cube(
            csi_data=csi_data,
            uav_rotation_l2w=np.eye(3),
            rcs_model=ConstantRCSModel(1.0),
            radar_system=radar,
            antenna_positions_m=ant_pos,
        )

        phase_step = np.angle(cube[0, 1, 0] / cube[0, 0, 0])
        expected = 2.0 * np.pi * 2.0 * csi_data["doppler"][0] * radar.effective_chirp_interval
        self.assertAlmostEqual(phase_step, expected, places=6)
        self.assertAlmostEqual(radar.idle_time, 0.25)
        np.testing.assert_allclose(radar.params_array(), [C_M_S, 4.0, 8.0, 1.0, 8.0, 1.25])

    def test_receiver_noise_is_deterministic_with_seeded_rng(self):
        radar = RadarSystem(
            f_c=C_M_S,
            bandwidth=4.0,
            sample_rate=8.0,
            chirp_duration=1.0,
            num_chirps=4,
            noise_floor_dbm=0.0,
        )
        csi_data = {
            "a": np.asarray([], dtype=np.complex128),
            "tau": np.asarray([], dtype=np.float64),
            "doppler": np.asarray([], dtype=np.float64),
            "theta_r": np.asarray([], dtype=np.float64),
            "phi_r": np.asarray([], dtype=np.float64),
            "theta_t": np.asarray([], dtype=np.float64),
            "phi_t": np.asarray([], dtype=np.float64),
        }
        ant_pos = virtual_array_positions(radar.f_c, shape=(1, 1), spacing_wavelengths=0.5)

        clean = synthesize_radar_cube(
            csi_data=csi_data,
            uav_rotation_l2w=np.eye(3),
            rcs_model=ConstantRCSModel(1.0),
            radar_system=radar,
            antenna_positions_m=ant_pos,
            add_noise=False,
        )
        noisy_a = synthesize_radar_cube(
            csi_data=csi_data,
            uav_rotation_l2w=np.eye(3),
            rcs_model=ConstantRCSModel(1.0),
            radar_system=radar,
            antenna_positions_m=ant_pos,
            add_noise=True,
            rng=np.random.default_rng(7),
        )
        noisy_b = synthesize_radar_cube(
            csi_data=csi_data,
            uav_rotation_l2w=np.eye(3),
            rcs_model=ConstantRCSModel(1.0),
            radar_system=radar,
            antenna_positions_m=ant_pos,
            add_noise=True,
            rng=np.random.default_rng(7),
        )

        np.testing.assert_allclose(clean, 0.0)
        np.testing.assert_allclose(noisy_a, noisy_b)
        self.assertFalse(np.allclose(clean, noisy_a))
        self.assertTrue(np.all(np.isfinite(noisy_a)))

    def test_spherical_wave_radar_uses_vertices_for_antenna_delays(self):
        radar = RadarSystem(
            f_c=C_M_S,
            bandwidth=4.0,
            sample_rate=8.0,
            chirp_duration=1.0,
            num_chirps=4,
        )
        csi_data = {
            "a": np.asarray([1.0 + 0.0j]),
            "tau": np.asarray([20.0 / C_M_S]),
            "doppler": np.asarray([0.0]),
            "theta_r": np.asarray([np.pi / 2.0]),
            "phi_r": np.asarray([0.0]),
            "theta_t": np.asarray([np.pi / 2.0]),
            "phi_t": np.asarray([np.pi / 2.0]),
            "tx_pos": np.asarray([0.0, 0.0, 0.0]),
            "uav_pos": np.asarray([10.0, 10.0, 0.0]),
            "interactions": np.asarray([[1]], dtype=np.int32),
            "vertices": np.asarray([[[0.0, 10.0, 0.0]]], dtype=np.float64),
            "path_interaction_count": np.asarray([1], dtype=np.int32),
        }
        ant_pos = virtual_array_positions(radar.f_c, shape=(1, 2), spacing_wavelengths=0.5)

        tau_ant = radar_one_way_tau_by_antenna(csi_data, ant_pos)
        np.testing.assert_allclose(tau_ant[:, 0], np.asarray([20.25, 19.75]) / C_M_S, atol=1e-12)

        cube = synthesize_radar_cube(
            csi_data=csi_data,
            uav_rotation_l2w=np.eye(3),
            rcs_model=ConstantRCSModel(1.0),
            radar_system=radar,
            antenna_positions_m=ant_pos,
            array_model="spherical-wave",
        )
        self.assertEqual(cube.shape, (2, 4, 8))
        self.assertEqual(cube.dtype, np.complex64)

    def test_compute_radar_maps_shapes(self):
        radar_params = np.asarray([C_M_S, 4.0, 8.0, 1.0, 8.0], dtype=np.float64)
        cube = np.ones((4, 4, 8), dtype=np.complex64)

        maps = compute_radar_maps(
            cube=cube,
            radar_params=radar_params,
            array_shape=(2, 2),
            angle_fft_size=8,
            remove_clutter=False,
        )

        self.assertEqual(maps["rd_db"].shape, (4, 4))
        self.assertEqual(maps["ra_db"].shape, (8, 4))
        self.assertEqual(maps["re_db"].shape, (8, 4))

    def test_range_map_peak_has_no_one_bin_offset(self):
        radar = RadarSystem(
            f_c=C_M_S,
            bandwidth=4.0,
            sample_rate=8.0,
            chirp_duration=1.0,
            num_chirps=4,
        )
        distance_m = C_M_S / 4.0
        csi_data = {
            "a": np.asarray([1.0 + 0.0j]),
            "tau": np.asarray([distance_m / C_M_S]),
            "doppler": np.asarray([0.0]),
            "theta_r": np.asarray([np.pi / 2.0]),
            "phi_r": np.asarray([0.0]),
            "theta_t": np.asarray([np.pi / 2.0]),
            "phi_t": np.asarray([0.0]),
        }
        cube = synthesize_radar_cube(
            csi_data=csi_data,
            uav_rotation_l2w=np.eye(3),
            rcs_model=ConstantRCSModel(1.0),
            radar_system=radar,
            antenna_positions_m=np.zeros((1, 3)),
        )
        maps = compute_radar_maps(
            cube,
            radar.params_array(),
            array_shape=(1, 1),
            angle_fft_size=8,
            remove_clutter=False,
        )
        _, range_idx = np.unravel_index(np.argmax(maps["rd_db"]), maps["rd_db"].shape)
        self.assertAlmostEqual(maps["range_axis_m"][range_idx], distance_m, places=5)


if __name__ == "__main__":
    unittest.main()
