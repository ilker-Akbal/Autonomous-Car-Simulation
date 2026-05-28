import time


class SimpleLogger:

    def __init__(self):
        self.start = time.time()

    def log(self, frame_id, summary):
        print(
            f"[{frame_id}] "
            f"action={summary['action']} "
            f"tl={summary['traffic_light_state']} "
            f"dist={summary['critical_car_distance']} "
            f"reason={summary['reason']}"
        )