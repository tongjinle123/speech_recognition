
from collections import defaultdict
import numpy as np
import torch as t


class MetricsManager(object):
    """
    MetricsManager
    """
    def __init__(self):
        self.metrics_val = defaultdict(float)
        self.metrics_cum = defaultdict(float)
        self.num_samples = 0

    def update(self, metrics):
        """
        update
        """
        num_samples = metrics.pop("num_samples", 1)
        self.num_samples += num_samples

        for key, val in metrics.items():
            if val is not None:
                if isinstance(val, t.Tensor):
                    val = val.item()
                    self.metrics_cum[key] += val * num_samples
                else:
                    assert len(val) == 2
                    val, num_words = val[0].item(), val[1]
                    self.metrics_cum[key] += np.array(
                        [val * num_samples, num_words])
                self.metrics_val[key] = val

    def clear(self):
        """
        clear
        """
        self.metrics_val = defaultdict(float)
        self.metrics_cum = defaultdict(float)
        self.num_samples = 0

    def get(self, name):
        """
        get
        """
        val = self.metrics_cum.get(name)
        if not isinstance(val, float):
            val = val[0]
        return val / self.num_samples

    def report_val(self):
        """
        report_val
        """
        metric_strs = []
        for key, val in self.metrics_val.items():
            metric_str = "{}:{:.3f}".format(key, val)
            metric_strs.append(metric_str)
        metric_strs = "   ".join(metric_strs)
        return metric_strs

    def report_cum(self):
        """
        report_cum
        """
        metric_strs = []
        for key, val in self.metrics_cum.items():
            if isinstance(val, float):
                val, num_words = val, None
            else:
                val, num_words = val

            metric_str = "{}:{:.3f}".format(key, val / self.num_samples)
            metric_strs.append(metric_str)

            if num_words is not None:
                ppl = np.exp(min(val / num_words, 100))
                metric_str = "{}_PPL:{:.3f}".format(key, ppl)
                metric_strs.append(metric_str)

        metric_strs = "   ".join(metric_strs)
        return metric_strs

    @staticmethod
    def extract(report):
        return {i.split(':')[0]: t.Tensor([float(i.split(':')[1])]) for i in report.split('   ')}
