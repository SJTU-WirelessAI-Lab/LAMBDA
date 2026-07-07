import os
import tempfile
import unittest

import numpy as np

from lambda_rf.utils.array_csi import (
    C_M_S,
    build_array_csi_fields,
    expand_csi_npz,
    load_rotation_matrix_from_pose_json,
    planar_array_positions,
    quaternion_xyzw_to_rotation_matrix,
)
from lambda_rf.tools.read_csi import load_csi_npz


def source_arrays(a_real, a_imag, theta_t, phi_t, theta_r, phi_r):
    return {
        "a_real": np.asarray(a_real, dtype=np.float32),
        "a_imag": np.asarray(a_imag, dtype=np.float32),
        "theta_t": np.asarray(theta_t, dtype=np.float32),
        "phi_t": np.asarray(phi_t, dtype=np.float32),
        "theta_r": np.asarray(theta_r, dtype=np.float32),
        "phi_r": np.asarray(phi_r, dtype=np.float32),
        "valid": np.ones_like(np.asarray(a_real, dtype=np.float32), dtype=bool),
    }


class ArrayCsiTest(unittest.TestCase):
    def test_one_by_one_array_matches_source_path_coefficients(self):
        arrays = source_arrays(
            a_real=[1.0, 2.0],
            a_imag=[0.5, -0.25],
            theta_t=[np.pi / 2, np.pi / 2],
            phi_t=[0.0, 0.0],
            theta_r=[np.pi / 2, np.pi / 2],
            phi_r=[0.0, 0.0],
        )

        fields = build_array_csi_fields(arrays, carrier_frequency_hz=C_M_S, tx_shape=(1, 1), rx_shape=(1, 1))
        mimo = fields["a_mimo_real"] + 1j * fields["a_mimo_imag"]

        np.testing.assert_allclose(mimo[0, 0, :], np.array([1.0 + 0.5j, 2.0 - 0.25j]), atol=1e-7)

    def test_receive_array_uses_far_field_phase(self):
        arrays = source_arrays(
            a_real=[1.0],
            a_imag=[0.0],
            theta_t=[np.pi / 2],
            phi_t=[0.0],
            theta_r=[np.pi / 2],
            phi_r=[np.pi / 2],
        )

        fields = build_array_csi_fields(arrays, carrier_frequency_hz=C_M_S, tx_shape=(1, 1), rx_shape=(1, 2))
        mimo = fields["a_mimo_real"] + 1j * fields["a_mimo_imag"]

        expected = np.array([-1j, 1j], dtype=np.complex64)
        np.testing.assert_allclose(mimo[:, 0, 0], expected, atol=1e-6)

    def test_spherical_wave_uses_path_vertices_for_tx_wavefront(self):
        arrays = source_arrays(
            a_real=[1.0],
            a_imag=[0.0],
            theta_t=[np.pi / 2],
            phi_t=[np.pi / 2],
            theta_r=[np.pi / 2],
            phi_r=[0.0],
        )
        arrays.update(
            {
                "tau": np.asarray([20.0 / C_M_S], dtype=np.float64),
                "tx_pos": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
                "uav_pos": np.asarray([10.0, 10.0, 0.0], dtype=np.float32),
                "interactions": np.asarray([[1]], dtype=np.int32),
                "path_vertices": np.asarray([[[0.0, 10.0, 0.0]]], dtype=np.float32),
                "path_interaction_count": np.asarray([1], dtype=np.int32),
            }
        )

        fields = build_array_csi_fields(
            arrays,
            carrier_frequency_hz=C_M_S,
            tx_shape=(1, 2),
            rx_shape=(1, 1),
            array_model="spherical-wave",
        )
        mimo = fields["a_mimo_real"] + 1j * fields["a_mimo_imag"]
        tau_mimo = fields["tau_mimo"]

        np.testing.assert_allclose(mimo[0, :, 0], np.array([-1j, 1j]), atol=1e-6)
        np.testing.assert_allclose(tau_mimo[0, :, 0], np.array([20.25, 19.75]) / C_M_S, atol=1e-12)
        self.assertEqual(str(fields["array_model"]), "spherical_wavefront_from_path_vertices")

    def test_spherical_wave_requires_compact_path_geometry_for_nlos_paths(self):
        arrays = source_arrays(
            a_real=[1.0],
            a_imag=[0.0],
            theta_t=[np.pi / 2],
            phi_t=[0.0],
            theta_r=[np.pi / 2],
            phi_r=[0.0],
        )
        arrays.update(
            {
                "tau": np.asarray([1e-7], dtype=np.float64),
                "tx_pos": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
                "uav_pos": np.asarray([10.0, 0.0, 0.0], dtype=np.float32),
                "interactions": np.asarray([[1]], dtype=np.int32),
                "path_interaction_count": np.asarray([1], dtype=np.int32),
            }
        )

        with self.assertRaisesRegex(KeyError, "path_vertices"):
            build_array_csi_fields(
                arrays,
                carrier_frequency_hz=C_M_S,
                tx_shape=(1, 1),
                rx_shape=(1, 1),
                array_model="spherical-wave",
            )

    def test_array_orientation_rotates_local_positions_into_world_frame(self):
        arrays = source_arrays(
            a_real=[1.0],
            a_imag=[0.0],
            theta_t=[np.pi / 2],
            phi_t=[0.0],
            theta_r=[np.pi / 2],
            phi_r=[0.0],
        )
        angle = np.pi / 2.0
        rotation = quaternion_xyzw_to_rotation_matrix([0.0, 0.0, np.sin(angle / 2.0), np.cos(angle / 2.0)])

        fields = build_array_csi_fields(
            arrays,
            carrier_frequency_hz=C_M_S,
            tx_shape=(1, 2),
            rx_shape=(1, 1),
            tx_rotation_matrix=rotation,
            tx_orientation_source="camera_pose.json",
        )

        local_positions = planar_array_positions((1, 2), wavelength_m=1.0)
        expected_world_positions = local_positions @ rotation.T
        np.testing.assert_allclose(fields["tx_array_pos"], expected_world_positions.astype(np.float32), atol=1e-7)
        np.testing.assert_allclose(fields["tx_array_rotation"], rotation.astype(np.float32), atol=1e-7)
        self.assertEqual(str(fields["tx_array_orientation_source"]), "camera_pose.json")

    def test_load_rotation_matrix_from_camera_pose_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pose_path = os.path.join(tmpdir, "BS1_Cam_world_pose.json")
            angle = np.pi / 2.0
            with open(pose_path, "w", encoding="utf-8") as f:
                f.write(
                    "{"
                    "\"world_transform\": {"
                    "\"orientation\": {"
                    f"\"w\": {np.cos(angle / 2.0)}, "
                    "\"x\": 0.0, "
                    "\"y\": 0.0, "
                    f"\"z\": {np.sin(angle / 2.0)}"
                    "}}}"
                )

            rotation = load_rotation_matrix_from_pose_json(pose_path)
            expected = quaternion_xyzw_to_rotation_matrix([0.0, 0.0, np.sin(angle / 2.0), np.cos(angle / 2.0)])
            np.testing.assert_allclose(rotation, expected, atol=1e-7)

    def test_expand_csi_npz_preserves_source_fields_and_adds_mimo(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "csi_000000.npz")
            output_path = os.path.join(tmpdir, "array", "csi_000000.npz")
            arrays = source_arrays(
                a_real=[1.0],
                a_imag=[0.0],
                theta_t=[np.pi / 2],
                phi_t=[0.0],
                theta_r=[np.pi / 2],
                phi_r=[0.0],
            )
            np.savez(input_path, **arrays, tau=np.array([1e-6]), doppler=np.array([0.0]), carrier_frequency=C_M_S)

            expand_csi_npz(input_path, output_path, tx_shape=(1, 1), rx_shape=(2, 2))

            with np.load(output_path, allow_pickle=False) as data:
                self.assertIn("a_real", data.files)
                self.assertIn("a_mimo_real", data.files)
                self.assertEqual(data["a_mimo_real"].shape, (4, 1, 1))
                np.testing.assert_array_equal(data["rx_array_shape"], np.array([2, 2], dtype=np.int32))

            summary = load_csi_npz(output_path)
            self.assertEqual(summary["mimo_shape"], (4, 1, 1))
            self.assertIn("mimo_path_power", summary)


if __name__ == "__main__":
    unittest.main()
