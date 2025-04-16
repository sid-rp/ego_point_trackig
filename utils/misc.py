import time
import torch
from collections import defaultdict

class SmoothedValue:
    """
    Keep track of a smoothed value over a window of last N values.
    """
    def __init__(self, window_size=20):
        self.window_size = window_size
        self.deque = []
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.total += value
        self.count += 1
        if len(self.deque) > self.window_size:
            removed_value = self.deque.pop(0)
            self.total -= removed_value
        return self

    @property
    def average(self):
        return self.total / len(self.deque) if len(self.deque) > 0 else 0.0

    @property
    def global_avg(self):
        return self.total / self.count if self.count > 0 else 0.0


class MetricLogger:
    """
    MetricLogger tracks multiple metrics and prints/logs them periodically.
    It supports both tracking smoothed values and printing to the terminal.
    """
    def __init__(self, delimiter="  ", window_size=20):
        self.meters = defaultdict(lambda: SmoothedValue(window_size))
        self.delimiter = delimiter
        self.start_time = time.time()

    def update(self, **kwargs):
        """
        Update metric values (e.g., loss, accuracy).
        """
        for name, value in kwargs.items():
            self.meters[name].update(value)


    def log_every(self, iterable, print_freq, header=None):
        """
        This function logs metrics every `print_freq` iterations.
        It supports any iterable (e.g., DataLoader).
        """
        header = header or ""
        total_iterations = len(iterable)  # Get total iterations in the epoch
        i = 0
        for obj in iterable:
            yield obj
            if i % print_freq == 0:
                # Print log summary after `print_freq` iterations
                self.print_metrics(i, header, total_iterations)
            i += 1
        # Print metrics one last time after finishing the loop
        self.print_metrics(i, header, total_iterations)


    def print_metrics(self, iteration, header="", total_iterations=None):
        """
        Print metrics in a human-readable format, including memory usage and iteration tracking.
        """
        elapsed_time = time.time() - self.start_time
        metrics_str = f"{header} | Iter [{iteration}/{total_iterations}] | Time {elapsed_time:.2f}s"
        
        for name, meter in self.meters.items():
            metrics_str += f"{self.delimiter} {name}: {meter.average:.4f} (Avg), {meter.global_avg:.4f} (Total)"

        if torch.cuda.is_available():
            mem_alloc = torch.cuda.memory_allocated() / 1024**2  # MB
            mem_reserved = torch.cuda.memory_reserved() / 1024**2  # MB
            metrics_str += f"{self.delimiter} Mem Alloc: {mem_alloc:.1f}MB"
            metrics_str += f"{self.delimiter} Mem Reserved: {mem_reserved:.1f}MB"

        print(metrics_str)

    def synchronize_between_processes(self):
        """
        Synchronizes metrics between different processes (useful for distributed training).
        """
        pass  # You can implement this based on your distributed setup


def custom_collate_fn(batch):
    from torch.nn.utils.rnn import pad_sequence

    def pad_keypoints_and_mask(kps_list):
        lengths = torch.tensor([kp.shape[0] for kp in kps_list], dtype=torch.long)
        padded = pad_sequence(kps_list, batch_first=True)  # [B, max_len, 2]
        max_len = padded.size(1)
        mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)  # [B, max_len]
        return padded, mask

    img1 = torch.stack([item['img1'] for item in batch])
    img2 = torch.stack([item['img2'] for item in batch])
    # dino_feats_img1 = torch.stack([item['img1_dino_feat'] for item in batch])
    # dino_feats_img2 = torch.stack([item['img2_dino_feat'] for item in batch])
    # F_matrix = torch.stack([item['F_matrix'] for item in batch])
    kp1_list = [item['kp1'] for item in batch]
    kp2_list = [item['kp2'] for item in batch]

    kp1_padded, kp1_mask = pad_keypoints_and_mask(kp1_list)
    kp2_padded, kp2_mask = pad_keypoints_and_mask(kp2_list)

    return {
        'img1': img1,
        'img2': img2,
        'kp1': kp1_padded,
        'kp2': kp2_padded,
        'kp1_mask': kp1_mask,
        'kp2_mask': kp2_mask,
        # "F_matrix": F_matrix
        # "img1_dino_feat": dino_feats_img1,
        # "img2_dino_feat": dino_feats_img2
    }
