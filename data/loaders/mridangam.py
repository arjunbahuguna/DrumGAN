from .base_loader import AudioDataLoader
from ..db_extractors.mridangam import extract


class Mridangam(AudioDataLoader):
    def load_data(self):
        self.data, self.metadata, self.header = \
            extract(self.data_path, self.criteria)