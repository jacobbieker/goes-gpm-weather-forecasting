# Run evaluation of the ERA5 forecast for all of 2020 and compare to the ERA5 reanalysis
import os.path

from utils import setup_era5_reanalysis, setup_era5_forecast, normalize_era5
import numpy as np

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--forecast_timesteps", type=int, default=8)
    parser.add_argument("--normalize_values", action="store_true")
    args = parser.parse_args()

    forecast_timesteps = args.forecast_timesteps
    normalize_values = args.normalize_values
    era5_dataset = setup_era5_reanalysis()
    era5_forecast = setup_era5_forecast(forecast_timesteps=forecast_timesteps)

    # Compare for the time steps and time periods in era5_forecast
    # Get the time steps in the forecast
    forecast_time_steps = era5_forecast.time.values
    # Get the time steps in the reanalysis
    reanalysis_time_steps = era5_dataset.time.values
    # Get the time steps in both
    time_steps = np.intersect1d(forecast_time_steps, reanalysis_time_steps)
    # Go per time in timestep and get the ERA5 forecast and reanalysis for each prediction_timedelta
    # Calculate the RMSE for each time step
    RMSEs = [[] for _ in range(forecast_timesteps)]
    temp_RMSE = [[] for _ in range(forecast_timesteps)]
    u_RMSE = [[] for _ in range(forecast_timesteps)]
    v_RMSE = [[] for _ in range(forecast_timesteps)]
    if normalize_values:
        tag = "normalized"
    else:
        tag = "unnormalized"
    if os.path.exists(f"RMSEs_{tag}.npy"):
        RMSEs = np.load(f"RMSEs_{tag}.npy", allow_pickle=True).tolist()
        # Get the number of time steps already done
        num_time_steps_done = len(RMSEs[0])
    else:
        num_time_steps_done = 0
    for idx, time_step in enumerate(time_steps):
        if idx < num_time_steps_done:
            continue
        for prediction_timestep in range(forecast_timesteps):
            # Get the ERA5 forecast
            forecast = era5_forecast.sel(time=time_step)
            valid_forecast_time = time_step + forecast.prediction_timedelta.values[prediction_timestep]
            assert valid_forecast_time == time_step + np.timedelta64(6 * prediction_timestep, "h")
            forecast = forecast.isel(
                prediction_timedelta=prediction_timestep
            )
            # Get the ERA5 reanalysis
            # Get the ERA5 reanalysis for the prediction_timedelta, which is the time of step * 6 hours
            reanalysis = era5_dataset.sel(
                time=time_step + np.timedelta64(6 * prediction_timestep, "h")
            )
            if normalize_values:
                forecast = normalize_era5(forecast)
                reanalysis = normalize_era5(reanalysis)
            forecast = forecast.to_dataarray().load()
            reanalysis = reanalysis.to_dataarray().load()
            # Calculate RMSE per variable
            for variable in forecast["variable"].values:
                # Calculate the RMSE
                rmse = np.sqrt(
                    np.mean(
                        (
                            forecast.sel(variable=variable).values
                            - reanalysis.sel(variable=variable).values
                        )
                        ** 2
                    )
                )
                # print(f"RMSE for {time_step} and {variable}: {rmse}")
                if variable == "2m_temperature":
                    temp_RMSE[prediction_timestep].append(rmse)
                elif variable == "10m_u_component_of_wind":
                    u_RMSE[prediction_timestep].append(rmse)
                elif variable == "10m_v_component_of_wind":
                    v_RMSE[prediction_timestep].append(rmse)
            # Calculate the RMSE
            rmse = np.sqrt(np.mean((forecast.values - reanalysis.values) ** 2))
            # print(f"RMSE for {time_step}: {rmse}")
            RMSEs[prediction_timestep].append(rmse)
        for i in range(forecast_timesteps):
            print(f"Average RMSE for {i}: {np.mean(RMSEs[i])}")
            print(f"Average temp RMSE for {i}: {np.mean(temp_RMSE[i])}")
            print(f"Average u RMSE for {i}: {np.mean(u_RMSE[i])}")
            print(f"Average v RMSE for {i}: {np.mean(v_RMSE[i])}")

        # Checkpoint calculations
        np.save(f"RMSEs_{tag}.npy", RMSEs)
        np.save(f"temp_RMSE_{tag}.npy", temp_RMSE)
        np.save(f"u_RMSE_{tag}.npy", u_RMSE)
        np.save(f"v_RMSE_{tag}.npy", v_RMSE)

    # Save RMSEs to disk
    np.save(f"RMSEs_{tag}.npy", RMSEs)
    np.save(f"temp_RMSE_{tag}.npy", temp_RMSE)
    np.save(f"u_RMSE_{tag}.npy", u_RMSE)
    np.save(f"v_RMSE_{tag}.npy", v_RMSE)
    # Print average RMSE
    for i in range(forecast_timesteps):
        print(f"Average RMSE for {i}: {np.mean(RMSEs[i])}")
        print(f"Average temp RMSE for {i}: {np.mean(temp_RMSE[i])}")
        print(f"Average u RMSE for {i}: {np.mean(u_RMSE[i])}")
        print(f"Average v RMSE for {i}: {np.mean(v_RMSE[i])}")

    print(f"Average RMSE: {np.mean(RMSEs)}")
    print(f"Average temp RMSE: {np.mean(temp_RMSE)}")
    print(f"Average u RMSE: {np.mean(u_RMSE)}")
    print(f"Average v RMSE: {np.mean(v_RMSE)}")
