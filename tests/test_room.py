import numpy as np

from app.geometry.room import fit_room, manhattan_angle, rotation_to_y


def synthetic_room(L=4.2, W=3.1, H=2.7, n=60000, seed=0, ceiling=True, rot_deg=23.0, tilt=True):
    """Points on floor, walls (and ceiling) of a box room, plus furniture clutter and noise."""
    rng = np.random.default_rng(seed)
    pts = []
    k = n // 6
    pts.append(np.c_[rng.uniform(0, L, k), np.zeros(k), rng.uniform(0, W, k)])  # floor
    if ceiling:
        pts.append(np.c_[rng.uniform(0, L, k // 2), np.full(k // 2, H), rng.uniform(0, W, k // 2)])
    for x in (0.0, L):
        pts.append(np.c_[np.full(k, x), rng.uniform(0, H, k), rng.uniform(0, W, k)])
    for z in (0.0, W):
        pts.append(np.c_[rng.uniform(0, L, k), rng.uniform(0, H, k), np.full(k, z)])
    pts.append(np.c_[rng.uniform(0.5, 1.5, k // 2), rng.uniform(0, 0.8, k // 2), rng.uniform(0.5, 1.2, k // 2)])  # a desk
    P = np.concatenate(pts) + rng.normal(0, 0.01, (sum(len(p) for p in pts), 3))
    a = np.deg2rad(rot_deg)
    Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    P = P @ Ry.T
    up = np.array([0.0, 1.0, 0.0])
    if tilt:  # arbitrary reconstruction frame: tilt the whole room
        t = np.deg2rad(17)
        Rx = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
        P, up = P @ Rx.T, Rx @ up
    return P, up


def test_rotation_to_y_maps_up():
    up = np.array([0.3, 0.8, -0.52])
    up /= np.linalg.norm(up)
    assert np.allclose(rotation_to_y(up) @ up, [0, 1, 0], atol=1e-9)


def test_manhattan_angle_recovers_rotation():
    rng = np.random.default_rng(0)
    xz = np.r_[np.c_[rng.uniform(0, 4, 3000), np.zeros(3000)], np.c_[np.zeros(3000), rng.uniform(0, 3, 3000)]]
    a = np.deg2rad(30)
    xz = xz @ np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]).T
    ang = manhattan_angle(xz)
    assert min(abs(ang - 30), abs(ang - 30 - 90), abs(ang - 30 + 90)) <= 1.5


def test_fit_room_on_synthetic_box_in_arbitrary_units():
    P, up = synthetic_room()
    units_per_m = 0.37  # the reconstruction is unscaled: 1 m = 0.37 units
    res = fit_room(P * units_per_m, up, k_cm_per_unit=100 / units_per_m, scale_method="test")
    assert abs(res.length_m - 4.2) / 4.2 < 0.03
    assert abs(res.width_m - 3.1) / 3.1 < 0.03
    assert abs(res.height_m - 2.7) / 2.7 < 0.03
    assert abs(res.floor_area_m2 - 4.2 * 3.1) / (4.2 * 3.1) < 0.05
    assert abs(res.wall_area_m2 - 2 * (4.2 + 3.1) * 2.7) / (2 * 7.3 * 2.7) < 0.05
    assert res.floor_area_ft2 and res.confidence > 0.5


def test_missing_ceiling_is_flagged():
    P, up = synthetic_room(ceiling=False)
    res = fit_room(P, up, k_cm_per_unit=100, scale_method="test")
    assert any("ceiling not captured" in n for n in res.notes)


def test_no_scale_means_no_numbers():
    P, up = synthetic_room()
    res = fit_room(P, up, k_cm_per_unit=None, scale_method="")
    assert res.floor_area_m2 is None and res.notes


def test_ensemble_reports_median_only_when_estimates_agree():
    from app.geometry.room import RoomResult, combine_estimates
    def est(L, W):
        return RoomResult(length_m=L, width_m=W, height_m=2.8, floor_area_m2=L * W, confidence=0.7)
    agree = combine_estimates([est(3.3, 2.8), est(3.4, 2.75), est(3.5, 2.9)], "t")
    assert agree.length_m == 3.4 and agree.floor_area_m2 and agree.confidence <= 0.9
    wild = combine_estimates([est(2.8, 2.5), est(3.6, 3.2), est(4.5, 4.4)], "t")
    assert wild.floor_area_m2 is None and any("unstable" in n for n in wild.notes)
