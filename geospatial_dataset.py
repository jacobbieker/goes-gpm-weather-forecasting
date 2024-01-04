from torch.utils.data import IterableDataset
from utils import (
    setup_imerg,
    get_imerg_images,
    get_goes_full_image,
    get_goes_image_cutout,
    setup_era5_reanalysis,
    get_era5_reanalysis,
    setup_era5_forecast,
)
import numpy as np
import datetime as dt
import logging
from glob import glob
import torch

_log = logging.getLogger(__name__)


class PreprocessedGeospatialDataset(IterableDataset):
    def __init__(
        self,
        folder: str,
        train: bool = True,
        fix_goes_scaling: bool = True,
        use_goes: bool = True,
        use_imerg: bool = True,
        use_fourier: bool = True,
        use_era5: bool = False,
    ):
        """
        Dataset using preprocessed batches of data, which are saved as PyTorch tensors.

        Args:
            folder: Folder where the data is located
            train: Whether this data is the training dataset or not
            fix_goes_scaling: Whether to fix the incorrect scaling of GOES data in the Hugging Face datasets
            use_goes: Whether to use GOES data
            use_imerg: Whether to use IMERG data
            use_fourier: Whether to use precomputed Fourier features based on the central lat/lon and time of the GOES imagery
            use_era5: Whether using ERA5 as input
        """
        super().__init__()
        self.train = train
        self.train_folder = folder
        self.train_files = glob(f"{folder}/*/*.pt")
        self.fix_goes_scaling = fix_goes_scaling
        self.use_goes = use_goes
        self.use_imerg = use_imerg
        self.use_fourier = use_fourier
        self.use_era5 = use_era5
        _log.info(f"Number of preprocessed batches: {len(self.train_files)}")

    def __iter__(self):
        if self.train:
            np.random.shuffle(self.train_files)
        for idx in range(len(self.train_files)):
            try:
                # Sometimes the file is corrupted, so skip it
                inputs, targets = torch.load(self.train_files[idx])
            except:
                continue
            # Select random example from the batch
            # NaNs in the input would be from going beyond Earth's disk in GOES or outside IMERGs view, so setting
            # to 0 is fine
            # Targets should never have that happen, so don't fill it in
            inputs[torch.isnan(inputs)] = 0
            # Fix GOES scaling, if needed, as the values are 4x too large on the HF data
            if self.fix_goes_scaling and self.use_goes:
                inputs[:, :16] /= 4
            # Get which channels to use
            channels_to_use = []
            if self.use_goes:
                channels_to_use.extend(list(range(16)))
            if self.use_fourier:
                channels_to_use.extend(list(range(16, 60)))
            if self.use_imerg:
                channels_to_use.extend(
                    [
                        60,
                    ]
                )
            if not self.use_era5:
                inputs = inputs[:, np.array(channels_to_use)]

            if self.train:
                example_idx = np.random.randint(0, inputs.shape[0])
                inputs, targets = inputs[example_idx], targets[example_idx]

            # Check for NaNs and infs
            if torch.any(torch.isnan(inputs)) or torch.any(torch.isnan(targets)):
                _log.debug("NaNs in inputs or targets, skipping")
                continue
            yield inputs, targets


class GeospatialDataset(IterableDataset):
    def __init__(
        self,
        image_size: int = 224,
        target_size: int = 16,
        num_frames: int = 4,
        use_imerg: bool = False,
        use_era5_input: bool = False,
        use_goes: bool = True,
        start_datetime: dt.datetime = dt.datetime(2017, 3, 1),
        end_datetime: dt.datetime = dt.datetime(2019, 12, 29),
        forecast_horizon_hours: int = 42,
        forecast_timesteps: int = 7,
        forecast_frequency_hours: int = 6,
        test: bool = False,
        **kwargs,
    ):
        """
        Create, combine, and return the input and target for the model, using a combination of GOES, IMERG, and ERA5.


        Args:
            image_size: Image size in pixels
            target_size: Target size in pixels
            num_frames: Number of history frames to include
            use_imerg: Whether to use IMERG data
            use_era5_input: Whether to use ERA5 data as input
            use_goes: Whether to use GOES data as input
            start_datetime: Start datetime to select from
            end_datetime: End datetime to select from
            forecast_horizon_hours: Forecast horizon in hours
            forecast_timesteps: Number of forecast timesteps to include
            forecast_frequency_hours: Frequency of forecast timesteps (i.e. every 6 hours, every hour, etc.)
            test: Whether to generate a test dataset using the ERA5 forecast from Weatherbench2
            **kwargs: start_val_idx: If building the test dataset, the index to start at if not starting from 0
        """
        super().__init__()
        if use_imerg:
            _log.info("Setting up IMERG dataset")
            self.imerg_dataset = setup_imerg()
        else:
            self.imerg_dataset = None
        _log.info("Setting up ERA5 Reanalysis dataset")
        self.era5_dataset = setup_era5_reanalysis()
        self.image_size = image_size
        self.target_size = target_size
        self.use_imerg = use_imerg
        self.use_era5_input = use_era5_input
        self.use_goes = use_goes
        self.num_frames = num_frames
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime
        self.forecast_horizon_hours = forecast_horizon_hours
        self.forecast_timesteps = forecast_timesteps
        self.forecast_frequency_hours = forecast_frequency_hours
        self.test = test
        # If testing, want to match the ERA5 forecast time steps, so have to load that
        if self.test:
            _log.info("Setting up ERA5 Forecast dataset")
            self.era5_forecast = setup_era5_forecast(forecast_timesteps=self.forecast_timesteps)
            self.start_val_idx = kwargs.get("start_val_idx", 0)

    def _get_t0_datetime(self):
        if self.test:
            all_init_times = self.era5_forecast.time.values
            for val_idx, t0_datetime in enumerate(all_init_times):
                if val_idx < self.start_val_idx:
                    continue
                # convert to Python datetime from np.datetime64
                t0_datetime = dt.datetime.utcfromtimestamp(t0_datetime.astype(int) * 1e-9)
                yield val_idx, t0_datetime
        else:
            idx = 0
            while True:
                t0_datetime = self.start_datetime + dt.timedelta(
                    seconds=np.random.randint(
                        0, int((self.end_datetime - self.start_datetime).total_seconds())
                    ),
                )
                # Round to the nearest hour
                t0_datetime = t0_datetime.replace(minute=0, second=0, microsecond=0)
                yield idx, t0_datetime
                idx += 1

    def __iter__(self):
        for idx, t0_datetime in self._get_t0_datetime():
            if self.use_goes:
                # Get the GOES image(s) for the given time period, and up to 1 hour before
                goes_image_stack = get_goes_full_image(
                    start_datetime=t0_datetime - dt.timedelta(hours=1),
                    end_datetime=t0_datetime,
                    image_type="FULL DISK",
                    max_images=self.num_frames,
                )
                # plot_inputs(goes_image_stack.isel(time=-1), era5=self.era5_dataset.sel(time=start_datetime))
                if goes_image_stack is None:
                    _log.debug("GOES image stack is None, skipping")
                    continue
                # Pick random x and y coordinates, which will be converted to lat/lon later
                max_lat = goes_image_stack["latitude"].max().values - 20
                min_lat = goes_image_stack["latitude"].min().values + 20
                max_lon = goes_image_stack["longitude"].max().values - 20
                min_lon = goes_image_stack["longitude"].min().values + 20

            elif self.use_era5_input:
                # Pick random x and y coordinates, which will be converted to lat/lon later
                max_lat = self.era5_dataset["latitude"].max().values - 20
                min_lat = self.era5_dataset["latitude"].min().values + 20
                max_lon = self.era5_dataset["longitude"].max().values - 20
                min_lon = self.era5_dataset["longitude"].min().values + 20

            center_lon = np.random.uniform(min_lon, max_lon, size=32)
            center_lat = np.random.uniform(min_lat, max_lat, size=32)
            for lat, lon in zip(center_lat, center_lon):
                # Get the GOES image(s) for the given time period, and up to 1 hour before
                if self.use_imerg and self.imerg_dataset is not None:
                    imerg_images: np.ndarray = get_imerg_images(
                        self.imerg_dataset,
                        [lat, lon],
                        end_datetime=t0_datetime,
                        max_images=self.num_frames,
                        image_size=self.image_size,
                    )
                if self.use_era5_input:
                    # ERA5 as input some cheats if within the last 3 hours, so take the ERA5 value for 3 hours before start_datetime as basis of input
                    era5_image: np.ndarray = get_era5_reanalysis(
                        self.era5_dataset,
                        [lat, lon],
                        start_datetime=t0_datetime - dt.timedelta(hours=12),
                        end_datetime=t0_datetime - dt.timedelta(hours=3),
                        normalize=True,
                        max_frames=self.num_frames,
                        image_size=self.image_size,
                    )
                    if (
                        era5_image.shape[-1] != self.image_size
                        or era5_image.shape[-2] != self.image_size
                        or era5_image.shape[1] != self.num_frames
                    ):
                        _log.debug("ERA5 input image is wrong size, skipping")
                        continue
                if self.use_goes:
                    goes_example_image: np.ndarray = get_goes_image_cutout(
                        goes_image_stack,
                        [lat, lon],
                        image_size=self.image_size,
                        add_fourier_features=True,
                        normalize=True,
                    )
                    if goes_example_image is None:
                        _log.debug("GOES image is in Space, skipping")
                        continue
                    if (
                        goes_example_image.shape[-1] != self.image_size
                        or goes_example_image.shape[-2] != self.image_size
                        or goes_example_image.shape[1] != self.num_frames
                    ):
                        _log.debug("GOES image is wrong size, skipping")
                        continue

                # Get the ERA5 ground truth for the target, which is from the 0th hour to the 6th hour
                era5_target: np.ndarray = get_era5_reanalysis(
                    self.era5_dataset,
                    [lat, lon],
                    start_datetime=t0_datetime,
                    end_datetime=t0_datetime + dt.timedelta(hours=self.forecast_horizon_hours),
                    time_resolution_hours=self.forecast_frequency_hours,
                    normalize=True,
                    image_size=16,
                    max_frames=self.forecast_timesteps,
                )
                # Check if image height and width are the correct size, skip if otherwise
                if (
                    era5_target.shape[-1] != self.target_size
                    or era5_target.shape[-2] != self.target_size
                ):
                    _log.debug("ERA5 target is wrong size, skipping")
                    continue

                model_input = []
                if self.use_goes:
                    model_input.append(np.array(goes_example_image, dtype=np.float32))
                if self.use_imerg:
                    model_input.append(np.array(imerg_images, dtype=np.float32))
                if self.use_era5_input:
                    model_input.append(np.array(era5_image, dtype=np.float32))

                # For the test dataset, want to know which validation index it is, so include it here
                x = model_input[0] if len(model_input) == 1 else np.concatenate(
                    model_input, axis=2
                )
                y = np.array(era5_target, dtype=np.float32)
                if self.test:
                    yield x, y, idx
                else:
                    yield x, y

            if self.use_goes:
                # Close xarray dataset of GOES imagery
                goes_image_stack.close()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    gdataset = GeospatialDataset(num_frames=2)
    for i, (model_input, target) in enumerate(gdataset):
        print(model_input.shape, target.shape)
        # Make a faceted plot of the input and target with one plot per channel
        fig, axes = plt.subplots(2, 3)
        axes[0, 0].imshow(model_input[0, 0])
        axes[0, 0].set_title("GOES Blue")
        axes[0, 1].imshow(model_input[0, 1])
        axes[0, 1].set_title("GOES, Red")
        axes[0, 2].imshow(model_input[0, 2])
        axes[0, 2].set_title("GOES, NIR")
        axes[1, 0].imshow(target[0, 0])
        axes[1, 0].set_title("ERA5 Temp")
        axes[1, 1].imshow(target[0, 1])
        axes[1, 1].set_title("ERA5 U Component")
        axes[1, 2].imshow(target[0, 2])
        axes[1, 2].set_title("ERA5 V Component")
        plt.show()
        fig.clf()
        if i > 2:
            break
