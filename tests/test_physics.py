import numpy as np

from surgecast import physics


def test_inverse_barometer_is_about_one_cm_per_hpa():
    assert abs(physics.inverse_barometer(np.array(1003.25)) - 0.0995) < 1e-3
    assert physics.inverse_barometer(np.array(physics.P_REF)) == 0


def test_wind_direction_convention():
    # a north wind blows FROM the north, i.e. towards the south
    u, v = physics.wind_components(np.array(10.0), np.array(0.0))
    assert abs(u) < 1e-9 and abs(v + 10) < 1e-9
    u, v = physics.wind_components(np.array(10.0), np.array(270.0))  # from the west
    assert abs(u - 10) < 1e-9 and abs(v) < 1e-9


def test_wind_stress_quadratic_and_saturating():
    tx, _ = physics.wind_stress(np.array([5.0, 10.0]), np.array([270.0, 270.0]))
    assert np.isclose(tx[1] / tx[0], 4.0)  # constant C_d below 11 m/s
    cd = physics.drag_coefficient(np.array([40.0, 60.0]))
    assert cd[0] == cd[1]  # capped in hurricane-force winds


def test_haversine_and_bearing():
    d = physics.haversine_km(0.0, 0.0, 0.0, 1.0)
    assert abs(d - 111.19) < 0.1
    assert np.isclose(physics.initial_bearing(0.0, 0.0, 1.0, 0.0), 0.0)  # due north
    assert np.isclose(physics.initial_bearing(0.0, 0.0, 0.0, 1.0), np.pi / 2)  # due east
