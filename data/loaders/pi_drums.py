from .base_loader import AudioDataLoader
from ..db_extractors.pi_drums import extract


class PIDrums(AudioDataLoader):
    def load_data(self):
        self.data, self.metadata, self.header = \
            extract(self.data_path, self.criteria)
