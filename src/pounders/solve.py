import copy
from typing import Tuple

import numpy as np
from pounders.solve_auxiliary import add_more_points
from pounders.solve_auxiliary import calc_res
from pounders.solve_auxiliary import compute_fnorm
from pounders.solve_auxiliary import find_nearby_points
from pounders.solve_auxiliary import get_params_quadratic_model
from pounders.solve_auxiliary import improve_model
from pounders.solve_auxiliary import solve_subproblem


def solve_pounders(
    x0: np.ndarray,
    nobs: int,
    f: callable,
    delta: float,
    delta_min: float,
    delta_max: float,
    gamma0: float,
    gamma1: float,
    theta1: float,
    theta2: float,
    eta0: float,
    eta1: float,
    c1: float,
    c2: int,
    gnorm_sub: float,
) -> Tuple[np.ndarray, np.ndarray]:
    n = x0.shape[0]  # number of model parameters
    maxinterp = 2 * n + 1  # max number of interpolation points

    xhist = np.zeros((1000, n))
    fhist = np.zeros((1000, nobs))
    fnorm = np.zeros(1000)
    fdiff = np.zeros((n, nobs))
    hess = np.zeros((nobs, n, n))
    model_indices = np.zeros(maxinterp, dtype=int)

    niter = 0
    last_mpoints = 0

    xhist[0] = x0
    fhist[0, :] = f(x0)
    fnorm[0] = compute_fnorm(fhist[0, :])

    minnorm = fnorm[0]
    minindex = 0

    # Increment parameters separately by delta
    for i in range(n):
        x1 = copy.deepcopy(x0)
        x1[i] += delta

        xhist[i + 1, :] = x1
        fhist[i + 1, :] = f(x1)
        fnorm[i + 1] = compute_fnorm(fhist[i + 1, :])

        if fnorm[i + 1] < minnorm:
            minnorm = fnorm[i + 1]
            minindex = i + 1

    xmin = xhist[minindex, :]
    fmin = fhist[minindex, :]
    fnorm_min = minnorm

    indices_not_min = [i for i in range(n + 1) if i != minindex]
    xk = (xhist[indices_not_min, :] - xmin) / delta
    fdiff = fhist[indices_not_min, :] - fmin

    # Determine the initial quadratic model
    fdiff = np.linalg.solve(xk, fdiff)

    jac_res = np.dot(fdiff, fmin)
    hess_res = np.dot(fdiff, fdiff.T)
    gnorm = np.linalg.norm(jac_res)
    gnorm *= delta

    valid = True
    reason = True
    nhist = n + 1
    mpoints = n + 1

    while reason is True:
        niter += 1

        # Solve the subproblem min{Q(s): ||s|| <= 1.0}
        rslt = solve_subproblem(jac_res, hess_res, gnorm_sub, n)

        qmin = -rslt.fun
        xhist[nhist, :] = xmin + rslt.x * delta
        fhist[nhist, :] = f(xhist[nhist, :])
        fnorm[nhist] = compute_fnorm(fhist[nhist, :])
        rho = (fnorm[minindex] - fnorm[nhist]) / qmin

        nhist += 1

        # Update the center
        if (rho >= eta1) or (rho > eta0 and valid is True):
            # Update model to reflect new base point
            x1 = (xhist[nhist, :] - xmin) / delta

            fdiff += np.dot(hess, x1).T
            fmin += 0.5 * np.dot(np.dot(x1, hess), x1) + np.dot(x1, fdiff)

            fnorm_min += np.dot(x1, jac_res) + 0.5 * np.dot(hess_res, x1)
            jac_res += np.dot(hess_res, x1)

            minindex = nhist - 1
            minnorm = fnorm[minindex]

            # Change current center
            xmin = xhist[minindex, :]

        # Evaluate at a model improving point if necessary
        # Note: valid is True in first iteration
        qmat = np.zeros((n, n))
        if valid is False:
            q_is_I = 1
            mpoints = 0
            qmat, model_indices, mpoints, q_is_I = find_nearby_points(
                xhist,
                xmin,
                delta,
                c1,
                nhist,
                theta1,
                model_indices,
                mpoints,
                n,
                q_is_I,
                qmat,
            )

            if mpoints < n:
                addallpoints = 1
                xhist, fhist, fnorm, model_indices, mpoints, nhist = improve_model(
                    xhist,
                    fhist,
                    fnorm,
                    jac_res,
                    hess_res,
                    qmat,
                    model_indices,
                    minindex,
                    mpoints,
                    addallpoints,
                    n,
                    nhist,
                    delta,
                    f,
                )

        # Update the trust region radius
        delta_old = delta
        norm_x_sub = np.sqrt(np.sum(rslt.x ** 2))

        if rho >= eta1 and norm_x_sub > 0.5 * delta:
            delta = min(delta * gamma1, delta_max)
        elif valid is True:
            delta = max(delta * gamma0, delta_min)

        # Compute the next interpolation set
        q_is_I = 1
        mpoints = 0
        qmat, model_indices, mpoints, q_is_I = find_nearby_points(
            xhist,
            xmin,
            delta,
            c1,
            nhist,
            theta1,
            model_indices,
            mpoints,
            n,
            q_is_I,
            qmat,
        )

        if mpoints == n:
            valid = True
        else:
            valid = False
            qmat, model_indices, mpoints, q_is_I = find_nearby_points(
                xhist,
                xmin,
                delta,
                c2,
                nhist,
                theta1,
                model_indices,
                mpoints,
                n,
                q_is_I,
                qmat,
            )

            if n > mpoints:
                # Model not valid. Add geometry points
                addallpoints = n - mpoints
                xhist, fhist, fnorm, model_indices, mpoints, nhist = improve_model(
                    xhist,
                    fhist,
                    fnorm,
                    jac_res,
                    hess_res,
                    qmat,
                    model_indices,
                    minindex,
                    mpoints,
                    addallpoints,
                    n,
                    nhist,
                    delta,
                    f,
                )

        model_indices[1 : mpoints + 1] = model_indices[:mpoints]
        mpoints += 1
        model_indices[0] = minindex

        L, Z, N, M, mpoints = add_more_points(
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
        )

        xk = (xhist[model_indices[:mpoints]] - xmin) / delta_old
        res = np.zeros((maxinterp, nobs))

        for j in range(nobs):
            workk = np.dot(xk, hess[j, :, :])

            for i in range(mpoints):
                res[i, j] = (
                    -fmin[j]
                    - np.dot(fdiff[:, j], xk[i, :])
                    - 0.5 * np.dot(workk[i, :], xk[i, :])
                    + fhist[model_indices[i], j]
                )

        jac_quadratic, hess_quadratic = get_params_quadratic_model(
            L, Z, N, M, res, mpoints, n, nobs
        )
        fdiff = jac_quadratic.T + (delta / delta_old) * fdiff
        hess = hess_quadratic + (delta / delta_old) * hess

        fmin = fhist[minindex]
        fnorm_min = fnorm[minindex]
        jac_res, hess_res = calc_res(fdiff, fmin, hess)

        solution = xhist[minindex, :]
        gradient = jac_res
        gnorm = np.linalg.norm(gradient)
        gnorm *= delta

        print(f"solution: {solution}")

        # Test for repeated model
        last_model_indices = np.zeros(maxinterp, dtype=int)
        if mpoints == last_mpoints:
            same = True
        else:
            same = False

        for i in range(mpoints):
            if same:
                if model_indices[i] == last_model_indices[i]:
                    same = True
                else:
                    same = False
            last_model_indices[i] = model_indices[i]

        last_mpoints = mpoints
        if (same is True) and (delta == delta_old):
            # Identical model used in successive iterations
            reason = False

    return solution, gradient
