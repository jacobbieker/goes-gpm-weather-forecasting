# Forecasting ERA5 from Satellite Imagery and GPM data

This project looks at using GOES-16/17/18 imagery and GPM precipication data
to forecast the next 6-42 hours of ERA5 surface temperature and wind component reanalysis data, from WeatherBench 2. The
forecasts are compared to the ERA5 model forecasts overall and for the 0th hour and 6th hour timestep, as those
are the only timesteps that are available through the WeatherBench 2 dataset, and makes for simpler comparison to the other
WeatherBench models.

The goal of this project is to show that direct observations can be used to forecast weather variables, without the need to
be initialized or given the analysis fields from a weather model, like most AI weather models are. While this is quite limited in 
scope at the moment, it is a first step towards a more general model that can be used to forecast weather variables from direct
observations, and could be used to forecast weather variables in areas where traditional weather models are not available, or to improve
the forecasts of weather models by using the direct observations as inputs to the ML weather model.

## Setup

Simply create the conda environment from the `environment.yml` file using `mamba` with

```bash
mamba env create -f environment.yml
conda activate weather_forecaast
```

## How To Run

### Data Preparation

The data can be streamed in from the cloud fairly easily, but to make it easier to run the model, the data was preprocessed and uploaded to Hugging Face.

To recreate that preprocessing, the following steps should be followed:

1. Your Planetary Computer API key should be set as an environment variable `PC_SDK_SUBSCRIPTION_KEY`.
2. Run the following command: 
3. 
```bash
python preprocess.py
```

Alternatively, the processing can be submitted to be ran on the cloud, after [configuring kbatch](https://kbatch.readthedocs.io/en/latest/index.html#configure), with the following command
```bash
kbatch job submit -f preprocess_data.yaml --output=name ---env="{\"HF_TOKEN\": \"$HF_TOKEN\"}"
```

This will start a VM in the Planetary Computer and upload data to Hugging Face. For faster processing, multiple jobs
can be submitted in a row. Command line options for the processing should be changed in `preprocess_data.yaml` if needed.

Once the data is pushed to Hugging Face, the data can be copied to local storage with the following command:
```bash
python download_from_hf.py
```

### Training

To run the training and test at the end on the test data (with the best performing model architecture by default), run the following command:
```bash
python run.py
```

Different options are available with the help, including changing where data is located, which data sources to include, etc.

### Evaluation of ERA5 Forecast

To get the evaluation metrics for the ERA5 forecast model, run the following command:
```bash
python evaluation.py
```

### Plot Examples

To plot two random examples, run the following command:
```bash
python plot.py
```

## Data

The data is primarily split between two sources: Planetery Computer for the GOES-16/17/18 imagery and GPM data, and
WeatherBench 2 for the ERA5 reanalysis and forecast data.

The GOES-16/17/18 imagery and GPM data is stored as Cloud-Optimized GeoTIFFs (COGs) in Azure, and is accessible through 
STAC API. This data is available from Feb 2017 through to the present, at between a 5 and 15 minute temporal resolution. Newer data is available in near real-time.

The GPM data is stored in a single Zarr in Azure, and is available from June 2000 through end of May 2021, and has global coverage
at 0.1 degree resolution and half-hourly timesteps. This data is generated from satellite observations, and rain gauge observations from around the world.
The product available in the Zarr store is the Final IMERG product, and so is the most accurate of the IMERG products, but is only available with a lag of a few months. 
For this small project, that is acceptable, but for a production system, the Early IMERG product would be more appropriate, as it is available with a lag of only a few hours.

The ERA5 reanalysis and forecasts are stored in Google Cloud Storage as a Zarr each, with global coverage at 0.25 degree resolution.
The ERA5 reanalysis is available in hourly timesteps from 1979 through to the present, and the ERA5 forecasts are made at midnight and noon UTC
for each day, with a 6 hour timestep. 

ERA5 is based on the operational ECMWF forecast and data assimilation model in 2016, and is kept consistent for the entire time period to provide a 
consistent assimilation process and forecasts across the multiple decades that ERA5 covers. As a reanalysis, ERA5 takes in observations from a variety of sources, including satellite imagery, and produces a best guess of the
state of the atmosphere at a given time, using observations from up to 3 hours into the future. The ERA5 forecasts are produced by running the ERA5 model forward in time.
New ERA5 reanalysis is available with a lag of 5 days.


## Approach

The approach taken is to use the GOES-16/17/18 imagery and GPM data to forecast the next 6-42 hours of ERA5 surface temperature and wind component reanalysis data with a transformer architecture, specifically a vision transformer.
The temperature and u/v component of the wind were chosen as they are fairly important variables for wind and solar generation, and there are a lot of observations of those variables, meaning the ERA5 reanalysis should be highly accurate 
for those variables compared to others, such as precipitation. The transformer architecture was chosen as it has had good results in other tasks and the attention mechanism should allow it to learn better what parts of the input images to focus on.

The forecast target is a random 16x16 ERA5 reanalysis target that falls within the GOES-16/17/18 imagery field of view. The inputs include
the last 4 images from GOES around the area, taken as a 224x224 patch of GOES imagery, covering up to the last hour of imagery and 448kmx448km area.
Optionally, GPM precipitation data was also included, including the last 4 frames (2 hours) of data, and covering a large area of 22.4 degrees around the central point. 
Additionally, a spatio-temporal embedding was optionally included as an input, which is based on Fourier features computed on the time of day, day of year, and latitude and longitude of the central target point.

For a baseline, the ERA5 forecast model was used. The forecast data only covers 2020, and has two model runs a day, at midnight and noon UTC, with 6 hour timesteps.
This means that the ERA5 forecast model is only available for every 6th hour timesteps, and so the model was only evaluated on those timesteps.
As a comparison, a model trained on a 4x64x64 patch of ERA5 reanalysis data was created as a second baseline comparison to the ERA5 forecast.

In following with the WeatherBench 2 deterministic scoring, RMSE was used as the loss function for each of the variables.

All the data can be streamed in from the Planetary Computer and Google Cloud quite easily, although because of local constraints (my internet is not super fast),
the Planetary Computer `kbatch` service was used to run processing scripts closer to the data in the cloud, and upload prepared batches to Hugging Face.
Each split is stored as a separate dataset in Hugging Face, and can be downloaded with the `download_from_hf.py` script. The reason for this, rather than having splits stored in a single
Hugging Face dataset, is that there ended up being large amounts of commits, which kept hitting Hugging Face's per-repo rate limits. By splitting each split into its own dataset, the rate limits
were mostly avoided.

All models were trained with the default settings in ```run.py```.

## Results

Here are a few tables of the best results for each combination of inputs and outputs.

For the models forecasting the next 6 hours of ERA5, the GOES+IMERG+Fourier model is still training at the moment:

------------------------------------------------------
| Model | Overall RMSE | Temp RMSE | U RMSE | V RMSE |
|-------|--------------|-----------|--------|--------|
| IMERG | 0.0820 | 0.0715 | 0.0806 | 0.0720 |
| GOES | 0.0791 | 0.0686 | 0.0765 | 0.0693 |
| GOES+IMERG | 0.0869 | 0.0908 | 0.0836 | 0.0735 |
| ERA5 | 0.0543 | 0.0242 | 0.0535 | 0.0622 |
| ERA5 Forecast | 0.1142 | 0.1493 | 0.1063 | 0.0937 |
------------------------------------------------------

------------------------------------------------------
| Model | Overall RMSE | Forecast Hour 0 | Forecast Hour 1 | Forecast Hour 2 | Forecast Hour 3 | Forecast Hour 4 | Forecast Hour 5 | Forecast Hour 6 |
|-------|--------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| IMERG | 0.0820 | 0.0830 | 0.0810 | 0.0808 | 0.0800 | 0.0809 | 0.0806 | 0.0807 | 
| GOES | 0.0791 | 0.0795 | 0.0774 | 0.0774 | 0.0776 | 0.0781 | 0.0778 | 0.0785 | 
| GOES+IMERG | 0.0869 | 0.0868 | 0.0868 | 0.0868 | 0.0868 | 0.0869 | 0.0869 | 0.0869 | 
| ERA5 | 0.0543 | 0.0545 | 0.0542 | 0.0538 | 0.0538 | 0.0539 | 0.0537 | 0.0534 | 
| ERA5 Forecast | 0.1142 | 0.1141 | N/A | N/A | N/A | N/A | N/A |  0.1142 |
------------------------------------------------------


Models forecasting from 6 hours to 42 hours into the future:

------------------------------------------------------
| Model | Overall RMSE | Temp RMSE | U RMSE | V RMSE |
|-------|--------------|-----------|--------|--------|
| GOES | 0.0941 | 0.0953 | 0.0855 | 0.0752 |
| IMERG | 0.0998 | 0.1109 | 0.0851 | 0.0692 |
| GOES+IMERG | 0.0802 | 0.0688 | 0.0771 | 0.0698 |
| GOES+IMERG+Fourier | 0.0876 | 0.0891 | 0.0772 | 0.0693 |
| ERA5 | 0.0461 | 0.0273 | 0.0487 | 0.0506 |
| ERA5 Forecast | 0.1142 | 0.1493 | 0.1064 | 0.0935 |
------------------------------------------------------

------------------------------------------------------
| Model | Overall RMSE | Forecast Hour 6 | Forecast Hour 12 | Forecast Hour 18 | Forecast Hour 24 | Forecast Hour 30 | Forecast Hour 36 | Forecast Hour 42 |
|-------|--------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| GOES | 0.0941 | 0.0927 | 0.0925 | 0.0934 | 0.0930 | 0.0937 | 0.0932 | 0.0933 | 
| IMERG | 0.0998 | 0.1002 | 0.0984 | 0.0989 | 0.0986 | 0.0990 | 0.0984 | 0.0982 | 
| GOES+IMERG | 0.0802 | 0.0800 | 0.0785 | 0.0788 | 0.0789 | 0.0787 | 0.0787 | 0.0794 | 
| GOES+IMERG+Fourier | 0.0876 | 0.0879 | 0.0863 | 0.0863 | 0.0853 | 0.0861 | 0.0867 | 0.0874 | 
| ERA5 | 0.0461 | 0.0379 | 0.0391 | 0.0419 | 0.0453 | 0.0479 | 0.0496 | 0.0506 | 
| ERA5 Forecast | 0.1142 | 0.1141 | 0.1142 | 0.1143 | 0.1141 | 0.1143 | 0.1141 | 0.1141 |
------------------------------------------------------


Interestingly, adding Fourier spatio-temporal features seems to make the model forecasts worse than without them.
For the long lead times (6-42 hours) combining the GOES and IMERG GPM data provides better results than either individually,
although does worse than either individual input for the 0-6 hour forecasts. GOES imagery alone does better than IMERG GPM data alone on forecasting
temperature at the surface, which makes sense as GOES is directly sensing the radiation coming from Earth, while IMERG is only looking at precipitation. On the other
hand, IMERG does better than GOES alone for the wind components on the 6-42 hour horizons, which is probably partly because the IMERG data covers a larger physical area in each example than the GOES imagery
and so can see where the precipitation is moving further away from the target point, and so can estimate the wind better further into the future. At the shorter lead times, GOES also beats IMERG for wind components,
which might be because of the higher resolution of the GOES imagery, and so the model can see the at a higher resolution for the shorter lead times.
As expected, the model trained with ERA5 reanalysis data performs the best, handily beating
any models just using observations, and the ERA5 forecast model as well. 

Looking at the RMSE per forecast hour, the results seem odd, with the loss not necessarily increasing as the forecast hour increases, which is what would
be expected. While this could be an issue with the evaluation code, the ERA5 forecast vs ERA5 reanalysis RMSE per forecast hour is quite similar, and there
are checks to ensure that the forecast hour is being calculated correctly. With more time, that is something to be investigated further.

Considering that the 3D ViT model is mostly unchanged, and just predicts an image instead of a single class, the results are quite good, and show that transformers
can be used to forecast weather variables from observations with relative accuracy. At the same time, there are quite a few limitations to this project, namely the
limited area and data used, and the limited number of variables forecasted. 


## Future Work

As part of some time constraints on this project, the following work was not completed, but would be quite interesting and useful to do:

* Use the Early Run GPM data as inputs to the model
* Gather the hourly timesteps of the ERA5 forecasts to better compare the model to the ERA5 forecasts
* Use Global Mosaic of Geostationary Satellite Images from NOAA to increase the spatial coverage to the rest of the world (preliminary data processing to Zarr is happening [here](https://huggingface.co/datasets/jacobbieker/global-mosaic-of-geostationary-images))
* Incorporate ASOS weather observations (available minutely over the US and with various delays elsewhere in the world)
* Give the model the raw surface and atmospheric point observations from NOAA as inputs
* Forecast more variables on pressure levels
* Investigate the odd RMSE per forecast hour results
* Investigate why Fourier features made the results worse
* Try this with global coverage of ERA5, so the model sees the whole world at once and predicts for the whole world
