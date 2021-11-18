from functools import partial
from typing import Dict
from typing import Tuple

import numpy as np
from scipy.linalg import qr_multiply
from scipy.optimize import Bounds
from scipy.optimize import minimize


def compute_fnorm(res: np.ndarray) -> np.ndarray:
    """Residual norm."""
    return np.dot(res, res)


def calc_res(
    fdiff: np.ndarray, fmin: np.ndarray, hess: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate residuals of the jacobian and hessian."""
    jac_res = np.dot(fdiff, fmin)
    hess_res = np.dot(fdiff, fdiff.T)

    dim_array = np.ones((1, hess.ndim), int).ravel()
    dim_array[0] = -1
    fmin_reshaped = fmin.reshape(dim_array)

    hess_res += np.sum(fmin_reshaped * hess, axis=0)

    return jac_res, hess_res


def solve_subproblem(
    jac_res: np.ndarray,
    hess_res: np.ndarray,
    gnorm: float,
    n: int,
) -> Dict[str, np.ndarray]:
    """Solve the subproblem."""
    x0 = np.zeros(n)

    # If no bounds are specified, use [-1, 1]
    bounds = Bounds(-np.ones(n), np.ones(n))

    evaluate_subproblem = partial(
        _evaluate_obj_and_grad, hess_res=hess_res, jac_res=jac_res
    )

    rslt = minimize(
        evaluate_subproblem,
        x0,
        method="trust-constr",
        jac=True,
        hess="2-point",
        bounds=bounds,
        options={"xtol": 1.0e-10, "gtol": gnorm},
    )

    return rslt


def find_nearby_points(
    xhist: np.ndarray,
    xmin: np.ndarray,
    delta: float,
    c: float,
    nhist: int,
    theta1: float,
    model_indices: np.ndarray,
    mpoints: int,
    n: int,
    q_is_I: int,
    qmat: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Find nearby points."""
    for i in range(nhist - 1, -1, -1):
        xk = (xhist[i, :] - xmin) / delta
        normd = np.linalg.norm(xk)

        xk_plus = xk

        if normd <= c:
            if q_is_I == 0:
                xk_plus, _ = qr_multiply(qmat, xk_plus)

            proj = np.linalg.norm(xk_plus[mpoints:])

            # Add this index to the model
            if proj >= theta1:
                qmat = np.zeros((n, n))
                model_indices[mpoints] = i
                mpoints += 1
                qmat[:, mpoints - 1] = xk
                q_is_I = 0

            if mpoints == n:
                break

    return qmat, model_indices, mpoints, q_is_I


def improve_model(
    xhist: np.ndarray,
    fhist: np.ndarray,
    fnorm: np.ndarray,
    jac_res: np.ndarray,
    hess_res: np.ndarray,
    qmat: np.ndarray,
    model_indices: np.ndarray,
    minindex: int,
    mpoints: int,
    addallpoints: int,
    n: int,
    nhist: int,
    delta: float,
    f: callable,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Improve the model"""
    minindex_internal = 0
    minvalue = np.inf
    work = np.zeros(3)

    qtmp, _ = qr_multiply(qmat, np.eye(3), mode="right")

    for i in range(mpoints, n):
        dp = np.dot(qtmp[:, i], jac_res)

        # Model says use the other direction!
        if dp > 0:
            qtmp[:, i] *= -1

        jac_res_new = jac_res + 0.5 * np.dot(hess_res, qtmp[:, i])
        work[i] = np.dot(qtmp[:, i], jac_res_new)

        if (i == mpoints) or (work[i] < minvalue):
            minindex_internal = i
            minvalue = work[i]

        if addallpoints != 0:
            xhist, fhist, fnorm, model_indices, mpoints, nhist = _add_point(
                xhist,
                fhist,
                fnorm,
                qtmp,
                model_indices,
                minindex,
                i,
                mpoints,
                nhist,
                delta,
                f,
            )

    if addallpoints == 0:
        xhist, fhist, fnorm, model_indices, mpoints, nhist = _add_point(
            xhist,
            fhist,
            fnorm,
            qtmp,
            model_indices,
            minindex,
            minindex_internal,
            mpoints,
            nhist,
            delta,
            f,
        )

    return xhist, fhist, fnorm, model_indices, mpoints, nhist


def add_more_points(
    xhist,
    xmin,
    model_indices,
    minindex,
    delta,
    c2,
    theta2,
    n,
    maxinterp,
    mpoints,
    nhist,
):
    """Add more points."""
    M = np.zeros((maxinterp, n + 1))
    N = np.zeros((maxinterp, int(n * (n + 1) / 2)))
    M[:, 0] = 1

    for i in range(n + 1):
        M[i, 1:] = (xhist[model_indices[i], :] - xmin) / delta
        N[i, :] = _evaluate_phi(M[i, 1:], n)

    # Now we add points until we have maxinterp starting with the most recent ones
    point = nhist - 1
    mpoints = n + 1

    while (mpoints < maxinterp) and (point >= 0):
        # Reject any points already in the model
        reject = 0

        for i in range(n + 1):
            if point == model_indices[i]:
                reject = 1
                break

        if reject == 0:
            workxvec = xhist[point]
            workxvec = workxvec - xhist[minindex]
            normd = np.linalg.norm(workxvec)
            normd /= delta

            if normd > c2:
                reject = 1

        else:
            point -= 1
            continue

        M[mpoints, 1:] = (xhist[point] - xmin) / delta
        N[mpoints, :] = _evaluate_phi(M[mpoints, 1:], n)

        Q_tmp = np.zeros((7, 7))
        Q_tmp[:7, : n + 1] = M

        L_tmp, _ = qr_multiply(
            Q_tmp[: mpoints + 1, :],
            N.T[: int(n * (n + 1) / 2), : mpoints + 1],
            mode="right",
        )
        beta = np.linalg.svd(L_tmp.T[n + 1 :], compute_uv=False)

        if beta[min(mpoints - n, int(n * (n + 1) / 2)) - 1] > theta2:
            # Accept point
            model_indices[mpoints] = point
            L = L_tmp

            mpoints += 1

        point -= 1

    cq, _ = qr_multiply(
        Q_tmp[:mpoints, :], np.eye(maxinterp)[:, :mpoints], mode="right"
    )
    Z = cq[:, n + 1 : mpoints]

    if mpoints == (n + 1):
        L = np.zeros((maxinterp, int(n * (n + 1) / 2)))
        L[:n, :n] = np.eye(n)

    return L, Z, N, M, mpoints


def get_params_quadratic_model(
    L: np.ndarray,
    Z: np.ndarray,
    N: np.ndarray,
    M: np.ndarray,
    res: np.ndarray,
    mpoints: int,
    n: int,
    nobs: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Get parameters of quadratic model.

    Computes the parameters of the quadratic model Q(x) = c + g'*x + 0.5*x*G*x'
    that satisfies the interpolation conditions Q(X[:,j]) = f(j)
    for j= 1,..., m and with a Hessian matrix of least Frobenius norm.
    """
    jac_quadratic = np.zeros((nobs, n))
    hess_quadratic = np.zeros((nobs, n, n))

    if mpoints == (n + 1):
        omega = np.zeros(n)
        beta = np.zeros(int(n * (n + 1) / 2))
    else:
        L_tmp = np.dot(L[:, n + 1 : mpoints].T, L[:, n + 1 : mpoints])

    for k in range(nobs):
        if mpoints != (n + 1):
            # Solve L'*L*Omega = Z' * RES_k
            omega = np.dot(Z[:mpoints, :].T, res[:mpoints, k])
            omega = np.linalg.solve(np.atleast_2d(L_tmp), np.atleast_1d(omega))

            beta = np.dot(np.atleast_2d(L[:, n + 1 : mpoints]), omega)

        rhs = res[:mpoints, k] - np.dot(N[:mpoints, :], beta)

        alpha = np.linalg.solve(M[: n + 1, : n + 1], rhs[: n + 1])
        jac_quadratic[k, :] = alpha[1 : (n + 1)]

        num = 0
        for i in range(n):
            hess_quadratic[k, i, i] = beta[num]
            num += 1
            for j in range(i + 1, n):
                hess_quadratic[k, j, i] = beta[num] / np.sqrt(2)
                hess_quadratic[k, i, j] = beta[num] / np.sqrt(2)
                num += 1

    return jac_quadratic, hess_quadratic


def _evaluate_obj_and_grad(
    x: np.ndarray, hess_res: np.ndarray, jac_res: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the objective and gradient of the subproblem."""
    grad = np.dot(hess_res, x)
    obj = 0.5 * np.dot(x, grad) + np.dot(jac_res, x)
    grad += jac_res

    return obj, grad


def _evaluate_phi(x: np.ndarray, n: int) -> np.ndarray:
    """Evaluate phi.

    Phi = .5*[x(1)^2  sqrt(2)*x(1)*x(2) ... sqrt(2)*x(1)*x(n) ...
        ... x(2)^2 sqrt(2)*x(2)*x(3) .. x(n)^2]
    """
    phi = np.zeros(int(n * (n + 1) / 2))

    j = 0
    for i in range(n):
        phi[j] = 0.5 * x[i] * x[i]
        j += 1

        for k in range(i + 1, n):
            phi[j] = x[i] * x[k] / np.sqrt(2)
            j += 1

    return phi


def _add_point(
    xhist: np.ndarray,
    fhist: np.ndarray,
    fnorm: np.ndarray,
    qtmp: np.ndarray,
    model_indices: np.ndarray,
    minindex: int,
    index: int,
    mpoints: int,
    nhist: int,
    delta: float,
    f: callable,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Add point."""
    # Create new vector in history: X[newidx] = X[index] + delta * X[index]
    xhist[nhist] = qtmp[:, index]
    xhist[nhist, :] = delta * xhist[nhist, :] + xhist[minindex]

    # Compute value of new vector
    res = f(xhist[nhist])
    fsum = compute_fnorm(res)
    fhist[nhist, :] = res
    fnorm[nhist] = fsum

    # Add new vector to the model
    model_indices[mpoints] = nhist
    mpoints += 1
    nhist += 1

    return xhist, fhist, fnorm, model_indices, mpoints, nhist
