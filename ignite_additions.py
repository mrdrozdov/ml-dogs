from ignite.engine import Engine, Events
from ignite.metrics import Metric


class CustomAccuracyMetric(Metric):

    required_output_keys = ('correct', 'total')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, output):
        correct, total = output
        self._correct += correct
        self._total += total

    def reset(self):
        self._correct = 0
        self._total = 0

    def compute(self):
        return self._correct / self._total

    def attach(self, engine, name, _usage=None):
        # restart every epoch
        engine.add_event_handler(Events.EPOCH_STARTED, self.started)
        # compute metric
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)
        # apply metric
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.completed, name)


class EpochIteratation(Metric):

    required_output_keys = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, output):
        self._iteration += 1

    def reset(self):
        self._iteration = 0

    def compute(self):
        return self._iteration

    def attach(self, engine, name, _usage=None):
        # restart every epoch
        engine.add_event_handler(Events.EPOCH_STARTED, self.started)
        # compute metric
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)
        # apply metric
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.completed, name)
