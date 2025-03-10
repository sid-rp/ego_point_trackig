"""
Modified code from https://github.com/nwojke/deep_sort
"""

from __future__ import absolute_import

import numpy as np
import torch

from . import linear_assignment
from .track import Track

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, cfg, metric, max_age=30, n_init=3, phalp_tracker=None, dims=None):
        self.cfg = cfg
        print('Random: ', self.cfg.random)
        self.metric = metric
        self.max_age = max_age
        self.n_init = n_init
        self.tracks = []
        self._next_id = 1
        self.tracked_cost = {}
        self.phalp_tracker = phalp_tracker
        self.gt_dict = {}
        self.idx_track = 0

        if (dims is not None):
            self.A_dim = dims[0]
            self.P_dim = dims[1]
            self.L_dim = dims[2]

    def predict(self):
        """Propagate track state distributions one time step forward.
        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.phalp_tracker, increase_age=True)

    def update(self, detections, frame_t, image_name):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """

        matches, unmatched_tracks, unmatched_detections, statistics = self._match(detections)
        self.tracked_cost[frame_t] = [statistics[0], matches, unmatched_tracks, unmatched_detections, statistics[1],
                                      statistics[2], statistics[3], statistics[4]]
        if (self.cfg.verbose): print(np.round(np.array(statistics[0]), 2))

        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(detections[detection_idx])

        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        for detection_idx in unmatched_detections:
            self.gt_dict[self.idx_track] = detections[detection_idx].detection_data['ground_truth']
            self._initiate_track(detections[detection_idx], detection_idx, self.idx_track)

            self.idx_track += 1

        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed() or t.is_tentative()]

        # Update features for each tracklet that is active
        appe_features, loca_features, pose_features, uv_maps, targets = [], [], [], [], []
        for track in self.tracks:
            if not (track.is_confirmed() or track.is_tentative()): continue
            if not self.cfg.use_pred:
                appe_features += [track.track_data['history'][-1]['appe']]
                loca_features += [track.track_data['history'][-1]['loca']]
                targets += [track.track_id]
            else:
                appe_features += [track.track_data['prediction']['appe'][-1]]
                loca_features += [track.track_data['prediction']['loca'][-1]]
                targets += [track.track_id]
        self.metric.partial_fit(np.asarray(appe_features), np.asarray(loca_features), np.asarray(targets), active_targets)

        return matches, statistics[-1]

    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            appe_emb = np.array([dets[i].detection_data['appe'] for i in detection_indices])
            loca_emb = np.array([dets[i].detection_data['loca'] for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance([appe_emb, loca_emb], targets, dims=[self.A_dim, self.L_dim],
                                               phalp_tracker=self.phalp_tracker)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed() or t.is_tentative()]
        self.id_to_index = {}

        for index, track_index in enumerate(confirmed_tracks):
            track_id = self.tracks[track_index].track_id
            self.id_to_index[track_id] = index
        # Associate confirmed tracks using appearance features.
        if not self.cfg.random:
            matches, unmatched_tracks, unmatched_detections, cost_matrix, cost_matrix_a = linear_assignment.matching_simple(
                gated_metric, self.metric.matching_threshold, self.max_age, self.tracks, detections, confirmed_tracks)
        else:
            matches, unmatched_tracks, unmatched_detections, cost_matrix, cost_matrix_a = linear_assignment.matching_simple_random(
                gated_metric, self.metric.matching_threshold, self.max_age, self.tracks, detections, confirmed_tracks)

        track_idt = [i for i, t in enumerate(self.tracks) if t.is_confirmed() or t.is_tentative()]
        detect_idt = [i for i, d in enumerate(detections)]

        return matches, unmatched_tracks, unmatched_detections, [cost_matrix, None, None, track_idt, detect_idt,
                                                                 cost_matrix_a]

    def _initiate_track(self, detection, detection_id, track_id):
        new_track = Track(self.cfg, track_id, self.n_init, self.max_age,
                          detection_data=detection.detection_data,
                          detection_id=detection_id,
                          dims=[self.A_dim, self.P_dim, self.L_dim])
        new_track.add_predicted()
        self.tracks.append(new_track)
        self._next_id += 1
