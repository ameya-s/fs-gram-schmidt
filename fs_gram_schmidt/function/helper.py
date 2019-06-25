import numpy as np


def cosine_sqr(X, y):
    numerator = (X.dot(y) ** 2)
    denominator = np.apply_along_axis(lambda x: x.dot(x), 1, X) * y.dot(y)
    denominator[np.where(denominator == 0)] = 1e-9

    cos_sqr_Xy = numerator / denominator

    return cos_sqr_Xy


def project_on_nullspace(feature_matrix, nullspace):
    nullspace_tr = nullspace.transpose()

    return np.dot(
        np.dot(nullspace, np.linalg.inv(np.dot(nullspace_tr, nullspace))),
        np.dot(nullspace_tr, feature_matrix.transpose())).transpose()


def _fac_term_odd(k):
    numerator = np.arange(2 * k - 1, 0, -2)
    twos = np.ones(k) * 2
    denominator = np.arange(k, 0, -1)
    denominator = twos * denominator
    q = (numerator / denominator)

    return np.prod(q)


def _fac_term_even(k):
    twos = np.ones(k) * 2
    numerator = np.arange(k, 0, -1)
    numerator = twos * numerator
    denominator = np.arange(2 * k + 1, 0, -2)

    numerator = np.append(numerator, 1)

    q = (numerator / denominator)

    return np.prod(q)


def CDF(cosine_sqr, space_dimension):
    x = cosine_sqr
    v = space_dimension

    if v >= 2:
        # Proceed
        if v % 2 == 0:
            # CDF for even stage
            Phi = lambda x, v: (1 + sum([(
                    _fac_term_even(k) * (1 - x) ** k) for k in range(
                1,
                int(v / 2 - 2) + 1)])) if v >= 6 else (1 if v == 4 else 0)

            Pv_x = lambda x, v: (2 / np.pi) * (np.arcsin(x ** 0.5) + (x * (
                    1 - x)) ** 0.5 * Phi(x, v))
        else:
            # CDF for odd

            Phi = lambda x, v: (1 + sum([(
                    _fac_term_odd(k) * (1 - x) ** k) for k in range(
                1,
                int((v - 3) / 2) + 1)])) if v >= 5 else 1

            Pv_x = lambda x, v: (x) ** 0.5 * Phi(x, v)

        return Pv_x(x, v)

    else:
        raise ('stage must be greater than or equal to 2')