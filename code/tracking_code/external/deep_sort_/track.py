"""
Modified code from https://github.com/nwojke/deep_sort
"""

import copy
from collections import deque

import numpy as np
import scipy.signal as signal
from scipy.ndimage.filters import gaussian_filter1d


class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    Mark this track as missed (no association at the current time step).
    """

    def __init__(self, cfg, track_id, n_init, max_age, detection_data=None, detection_id=None, dims=None):
        self.cfg = cfg
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.time_init = detection_data["time"]
        self.state = TrackState.Tentative

        self._n_init = n_init
        self._max_age = max_age

        if (dims is not None):
            self.A_dim = dims[0]
            self.P_dim = dims[1]
            self.L_dim = dims[2]

        self.track_data = {"history": deque(maxlen=self.cfg.track_history), "prediction": {}}
        for _ in range(self.cfg.track_history):
            self.track_data["history"].append(detection_data)

    def predict(self, phalp_tracker, increase_age=True):
        if (increase_age):
            self.age += 1;
            self.time_since_update += 1

    def add_predicted(self, appe=None, pose=None, loca=None, uv=None):
        if not self.cfg.use_pred:
            pass
        else:
            appe_predicted = copy.deepcopy(appe.numpy()) if (appe is not None) else copy.deepcopy(
                self.track_data['history'][-1]['appe'])
            loca_predicted = copy.deepcopy(loca.numpy()) if (loca is not None) else copy.deepcopy(
                self.track_data['history'][-1]['loca'])

            self.track_data['prediction']['appe'].append(appe_predicted)
            self.track_data['prediction']['loca'].append(loca_predicted)

    def update(self, detection):
        self.track_data["history"].append(copy.deepcopy(detection.detection_data))
        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted

    def smooth_bbox(self, bbox):
        kernel_size = 5
        sigma = 3
        bbox = np.array(bbox)
        smoothed = np.array([signal.medfilt(param, kernel_size) for param in bbox.T]).T
        out = np.array([gaussian_filter1d(traj, sigma) for traj in smoothed.T]).T
        return list(out)
