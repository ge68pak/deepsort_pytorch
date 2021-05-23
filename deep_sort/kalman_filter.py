# vim: expandtab:ts=4:sw=4
import numpy as np
import scipy.linalg


"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).

    """

    def __init__(self):
        ndim, dt = 4, 1.

        # Create Kalman filter model matrices.
        # motion_mat =A_matrix
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)    # A 8*8
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)        # H 4*8

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):  # measurement y -> mean x(k-1)  covariance P(k-1)
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        mean_pos = measurement  # y 4*1
        mean_vel = np.zeros_like(mean_pos) # 4*1 zero vector
        mean = np.r_[mean_pos, mean_vel]   # 8*1 [x y a h 0 0 0 0] x(k-1)

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]]  # 8*1 vector
        covariance = np.diag(np.square(std))  # 8*8   P(k-1) every elements square and put in diag
        return mean, covariance   # x(k-1) P(k-1)

    def predict(self, mean, covariance): # update  x(k-1) P(k-1) -> x(k) P(k)
        """Run Kalman filter prediction step.
        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]]     # 4*1 vector
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]]    # 4*1 vector
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))    # Q 8*8
        #kalman滤波公式1,2 x(k)=Ax(k-1)   P(k)=AP(k-1)A'+Q
        mean = np.dot(self._motion_mat, mean)         # x(k)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov    # P(k)

        return mean, covariance   #  x(k) P(k)

    def project(self, mean, covariance):   # project: x(k) P(k) -> Hx(k) HP(k)H' + R
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]       # 4*1 vector
        innovation_cov = np.diag(np.square(std))       # R 4*4

        mean = np.dot(self._update_mat, mean)          # 4*8*8*1=4*1 将均值向量映射到检测空间  Hx(k)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))  # 4*8*8*8*8*4=4*4 将协方差矩阵映射到检测空间 HP(k)H'
        return mean, covariance + innovation_cov   # Hx(k), HP(k)H' + R

    def update(self, mean, covariance, measurement): # update: x(k),P(k),y -> x(k),P(k)
        """Run Kalman filter correction step.
            # 通过估计值和观测值估计最新结果
        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """
        # 将均值和协方差映射到检测空间，得到 Hx(k) 和 HP(k)H'
        projected_mean, projected_cov = self.project(mean, covariance)
        # 4*1 Hx(k)     4*4 HP(k)H'                  8*1 x(k)   8*8 P(k)
        #计算矩阵的Cholesky分解
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        #kalman滤波公式3 K(k)=P(k)H'(HP(k)H'+R)^-
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean    # y - Hx(k)
        # kalman滤波公式4 x(k)=x(k)+K(k)[y-Hx(k)]
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        # kalman滤波公式5 P(k)=P(k)-K(k)HP(k)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance    # x(k),P(k)

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False):
        """Compute gating distance between state distribution and measurements.

        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.

        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.

        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.

        """
        mean, covariance = self.project(mean, covariance)
        # 4*1 Hx(k)     4*4 HP(k)H'     8*1 x(k)   8*8 P(k)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean        # N*4
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False,
            overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)   #N*1
        return squared_maha
