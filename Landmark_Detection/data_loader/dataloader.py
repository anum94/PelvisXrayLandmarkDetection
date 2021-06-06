from base import BaseDataLoader
from data.Xray_dataset import Dataset


class XRayDataLoader(BaseDataLoader):
    """
    XRay data loading using BaseDataLoader
    """

    def __init__(
        self,
        data_dir,
        batch_size,
        shuffle=True,
        validation_split=0.0,
        num_workers=1,
        num_samples_to_load=1,
        limit_samples=False,
        training=True,
    ):

        self.data_dir = data_dir
        self.num_samples_to_load = num_samples_to_load
        self.limit_samples = limit_samples

        self.dataset = Dataset(
            self.data_dir, self.limit_samples, self.num_samples_to_load
        )
        super(XRayDataLoader, self).__init__(
            self.dataset,
            batch_size,
            shuffle,
            validation_split,
            num_workers,
            limit_samples,
            num_samples_to_load,
        )
