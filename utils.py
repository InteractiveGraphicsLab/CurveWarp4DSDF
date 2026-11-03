import numpy as np
import scipy.ndimage as nd
from scipy.ndimage import binary_erosion


def compute_exact_sdf(binary_mask_3d: np.ndarray, voxel_spacing: tuple) -> np.ndarray:
    mask = binary_mask_3d.astype(bool)
    if not np.any(mask):
        return np.ones_like(binary_mask_3d, dtype=np.float32) * 100.0

    dist_out = nd.distance_transform_edt(~mask, sampling=voxel_spacing)
    dist_in = nd.distance_transform_edt(mask, sampling=voxel_spacing)
    
    sdf = dist_out - dist_in
    return sdf.astype(np.float32)


import numpy as np


def interpolate_curve(cps: np.ndarray, num_points: int = 100) -> np.ndarray:
    cps = np.asarray(cps, dtype=np.float64)

    if cps.ndim != 2 or cps.shape[1] not in (2, 3):
        raise ValueError("cps must have shape (N, 2) or (N, 3)")

    if len(cps) < 2:
        return cps.copy()

    if num_points < 2:
        raise ValueError("num_points must be >= 2")

    original_dim = cps.shape[1]

    if original_dim == 3:

        stds = np.std(cps, axis=0)

        fixed_axis = int(np.argmin(stds))

        active_axes = [
            i for i in range(3)
            if i != fixed_axis
        ]

        pts_2d = cps[:, active_axes].copy()
        fixed_value = cps[0, fixed_axis]

    else:

        pts_2d = cps.copy()

        fixed_axis = None
        active_axes = None
        fixed_value = None

    bezier_cps = _calc_bezier_cps(
        pts_2d,
        number_iteration=30
    )

    if bezier_cps is None:
        print("error bezier_cps is None\n")
        return np.empty((0, original_dim), dtype=np.float64)

    curves_2d = _calc_bezier_curves(
        bezier_cps,
        pts_2d,
        num_points
    )

    if len(curves_2d) == 0:
        print("error curves_2d == 0\n")
        return np.empty((0, original_dim), dtype=np.float64)

    if original_dim == 3:

        curves_3d = np.zeros(
            (len(curves_2d), 3),
            dtype=np.float64
        )

        curves_3d[:, active_axes] = curves_2d
        curves_3d[:, fixed_axis] = fixed_value

        return curves_3d

    return curves_2d


def _calc_bezier_cps(kcurves_cps: np.ndarray,number_iteration: int = 30) -> np.ndarray:

    P = np.asarray(kcurves_cps, dtype=np.float64)

    n_points = len(P)

    number_segments = n_points - 2

    if number_segments <= 0:
        return None

    bezier_cps = np.zeros(
        (2 * number_segments + 1, 2),
        dtype=np.float64
    )

    lam = np.zeros(
        number_segments + 1,
        dtype=np.float64
    )

    t = np.zeros(
        number_segments + 2,
        dtype=np.float64
    )

    bezier_cps[0] = P[0]

    bezier_cps[2 * number_segments] = P[
        number_segments + 1
    ]

    lam[0] = 0.0
    lam[number_segments] = 1.0

    t[0] = 0.0
    t[number_segments + 1] = 1.0

    for i in range(number_segments):

        bezier_cps[2 * i + 1] = P[i + 1]

        t[i + 1] = 0.5

    for i in range(number_segments - 1):

        bezier_cps[2 * i + 2] = (
            bezier_cps[2 * i + 1]
            + bezier_cps[2 * i + 3]
        ) / 2.0

        lam[i + 1] = 0.5

    for _ in range(number_iteration):

        for n in range(number_segments - 1):

            i = n + 1

            a = np.sqrt(
                max(
                    _calc_area(
                        bezier_cps[2 * i - 2],
                        bezier_cps[2 * i - 1],
                        bezier_cps[2 * i + 1]
                    ),
                    0.0
                )
            )

            b = np.sqrt(
                max(
                    _calc_area(
                        bezier_cps[2 * i - 1],
                        bezier_cps[2 * i + 1],
                        bezier_cps[2 * i + 2]
                    ),
                    0.0
                )
            )

            if a + b > 1e-12:
                lam[i] = a / (a + b)
            else:
                lam[i] = 0.5


        for n in range(number_segments - 1):

            i = n + 1

            bezier_cps[2 * i] = (
                (1.0 - lam[i])
                * bezier_cps[2 * i - 1]
                + lam[i]
                * bezier_cps[2 * i + 1]
            )

        for n in range(number_segments):

            i = n + 1

            vec = (
                bezier_cps[2 * i]
                - bezier_cps[2 * i - 2]
            )

            diff = (
                P[i]
                - bezier_cps[2 * i - 2]
            )

            a = np.dot(vec, vec)

            b = (
                -3.0
                * np.dot(vec, diff)
            )

            c = np.dot(
                -3.0 * bezier_cps[2 * i - 2]
                + 2.0 * P[i]
                + bezier_cps[2 * i],
                diff
            )

            d = -np.dot(diff, diff)

            t[i] = _solve_formula(
                a, b, c, d
            )

        matrix_size = number_segments + 4

        mat_A = np.zeros(
            (matrix_size, matrix_size),
            dtype=np.float64
        )

        vec_bx = np.zeros(
            matrix_size,
            dtype=np.float64
        )

        vec_by = np.zeros(
            matrix_size,
            dtype=np.float64
        )

        mat_A[0, 0] = 1.0
        mat_A[1, 1] = 1.0

        mat_A[
            number_segments + 2,
            number_segments + 2
        ] = 1.0

        mat_A[
            number_segments + 3,
            number_segments + 3
        ] = 1.0


        vec_bx[0] = (
            bezier_cps[2 * number_segments, 0]
        )

        vec_by[0] = (
            bezier_cps[2 * number_segments, 1]
        )

        vec_bx[1] = P[0, 0]
        vec_by[1] = P[0, 1]

        vec_bx[number_segments + 2] = (
            P[number_segments + 1, 0]
        )

        vec_by[number_segments + 2] = (
            P[number_segments + 1, 1]
        )

        vec_bx[number_segments + 3] = (
            bezier_cps[0, 0]
        )

        vec_by[number_segments + 3] = (
            bezier_cps[0, 1]
        )

        for n in range(number_segments):

            i = n + 1

            ti = t[i]

            a = (
                (1.0 - lam[i - 1])
                * (1.0 - ti) ** 2
            )

            b = (
                lam[i - 1]
                * (1.0 - ti) ** 2
                + (
                    2.0
                    - (1.0 + lam[i]) * ti
                ) * ti
            )

            c = (
                lam[i]
                * ti ** 2
            )

            mat_A[i + 1, i] = a
            mat_A[i + 1, i + 1] = b
            mat_A[i + 1, i + 2] = c

            vec_bx[i + 1] = P[i, 0]
            vec_by[i + 1] = P[i, 1]

        try:
            vec_xx = np.linalg.solve(
                mat_A,
                vec_bx
            )

            vec_xy = np.linalg.solve(
                mat_A,
                vec_by
            )

        except np.linalg.LinAlgError:

            vec_xx = np.linalg.lstsq(
                mat_A,
                vec_bx,
                rcond=None
            )[0]

            vec_xy = np.linalg.lstsq(
                mat_A,
                vec_by,
                rcond=None
            )[0]

        for i in range(
            1,
            number_segments + 1
        ):

            bezier_cps[
                2 * i - 1,
                0
            ] = vec_xx[i + 1]

            bezier_cps[
                2 * i - 1,
                1
            ] = vec_xy[i + 1]

    return bezier_cps




def _calc_area(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray
) -> float:

    vec_a = b - a
    vec_b = c - a

    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    product_norm_ab = norm_a * norm_b

    if product_norm_ab < 1e-12:
        return 0.0

    theta = (
        np.dot(vec_a, vec_b)
        / product_norm_ab
    )

    theta = np.clip(
        theta,
        -1.0,
        1.0
    )

    area = (
        product_norm_ab
        * np.sqrt(
            max(
                0.0,
                1.0 - theta * theta
            )
        )
    )

    return float(area)


def _solve_formula(
    a: float,
    b_: float,
    c_: float,
    d_: float
) -> float:

    if abs(a) < 1e-12:
        return 0.5

    b = b_ / a / 3.0
    c = c_ / a
    d = d_ / a


    p = c / 3.0 - b * b

    q = (
        b * b * b
        - (b * c - d) / 2.0
    )

    D = q * q + p * p * p


    if abs(D) < np.finfo(np.float32).tiny:

        i = np.cbrt(q) - b

        if i >= 0.0:

            return min(
                i,
                1.0
            )

        else:

            return min(
                i * -2.0,
                1.0
            )


    elif D > 0.0:

        sqrt_D = np.sqrt(D)

        u = np.cbrt(
            -q + sqrt_D
        )

        v = np.cbrt(
            -q - sqrt_D
        )

        i = u + v - b

        if i < 0.0:
            return 0.0

        if i > 1.0:
            return 1.0

        return float(i)

    else:

        n = 2.0 * np.sqrt(-p)

        arg = (
            np.arctan2(
                np.sqrt(-D),
                -q
            ) / 3.0
        )

        pi2d3 = 2.0 * np.pi / 3.0

        i = (
            n * np.cos(arg)
            - b
        )

        if 0.0 <= i <= 1.0:
            return float(i)

        j = (
            n * np.cos(
                arg + pi2d3
            )
            - b
        )

        if 0.0 <= j <= 1.0:
            return float(j)

        k = (
            n * np.cos(
                arg - pi2d3
            )
            - b
        )

        if 0.0 <= k <= 1.0:
            return float(k)


    return 0.50001


def _calc_bezier_curves(
    bezier_cps: np.ndarray,
    kcurves_cps: np.ndarray,
    num_points: int
) -> np.ndarray:

    number_segments = len(bezier_cps) // 2

    if number_segments == 1:

        samples_per_segment = num_points

    else:

        samples_per_segment = max(
            2,
            int(
                np.ceil(
                    (num_points - 1)
                    / number_segments
                )
            ) + 1
        )

    curves = []

    for i in range(number_segments):

        p0 = bezier_cps[2 * i]
        p1 = bezier_cps[2 * i + 1]
        p2 = bezier_cps[2 * i + 2]

        ts = np.linspace(
            0.0,
            1.0,
            samples_per_segment
        )

        for j, t in enumerate(ts):
            p = (
                (1.0 - t) ** 2 * p0
                + 2.0 * (1.0 - t) * t * p1
                + t ** 2 * p2
            )

            if i > 0 and j == 0:
                continue

            curves.append(p)

    curves = np.asarray(
        curves,
        dtype=np.float64
    )

    if len(curves) == num_points:
        return curves

    if len(curves) < 2:
        return curves

    lengths = np.linalg.norm(
        np.diff(curves, axis=0),
        axis=1
    )

    cumulative = np.concatenate([
        [0.0],
        np.cumsum(lengths)
    ])

    total_length = cumulative[-1]

    if total_length < 1e-12:

        return np.repeat(
            curves[:1],
            num_points,
            axis=0
        )

    target = np.linspace(
        0.0,
        total_length,
        num_points
    )

    result = np.column_stack([
        np.interp(
            target,
            cumulative,
            curves[:, dim]
        )
        for dim in range(2)
    ])

    return result


def _is_inside_triangle(
    p: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray
) -> bool:
    ab = b - a
    bc = c - b
    ca = a - c

    bp = p - b
    cp = p - c
    ap = p - a

    abp = (
        ab[0] * bp[1]
        - ab[1] * bp[0]
    )

    bcp = (
        bc[0] * cp[1]
        - bc[1] * cp[0]
    )

    cap = (
        ca[0] * ap[1]
        - ca[1] * ap[0]
    )


    if (
        (abp > 0.0 and
         bcp > 0.0 and
         cap > 0.0)
        or
        (abp < 0.0 and
         bcp < 0.0 and
         cap < 0.0)
    ):
        return True

    return False


def extract_boundary_samples(mask_3d: np.ndarray, num_samples: int = 150) -> np.ndarray:
    eroded = binary_erosion(mask_3d)
    boundary = mask_3d & (~eroded)
    z, y, x = np.where(boundary)
    
    if len(z) == 0:
        return np.empty((0, 3))
        
    num_pts = min(num_samples, len(z))
    chosen = np.random.choice(len(z), size=num_pts, replace=False)
    return np.stack([x[chosen], y[chosen], z[chosen]], axis=1)