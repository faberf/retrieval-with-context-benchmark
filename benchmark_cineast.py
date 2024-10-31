import time
from collections import deque
from datetime import datetime
import math
import os

# Configuration
JSON_FILE = "features/features_visualtextcoembedding.json"
OUTPUT_FILE = "cineast_benchmark_log.txt"
MEASUREMENT_WINDOW = 10000  # Number of measurements to store

def count_items():
    """Counts the number of items in the JSON file by counting non-empty lines."""
    try:
        if not os.path.exists(JSON_FILE):
            print(f"File {JSON_FILE} does not exist yet. Waiting for the file to be created...")
            return 0  # Return 0 items if the file does not exist
        with open(JSON_FILE, 'r') as f:
            lines = f.readlines()
            return sum(1 for line in lines if line.strip() and not line.strip().startswith('['))
    except IOError as e:
        print(f"Error reading file: {e}")
        return 0  # Return 0 to indicate no items

def calculate_metrics(measurements):
    """Calculates average rates and standard deviation of insertion times."""
    if len(measurements) < 2:
        return None  # Not enough data to compute metrics

    # Total count and time differences between first and last measurements
    first_time, first_count = measurements[0]['time'], measurements[0]['count']
    last_time, last_count = measurements[-1]['time'], measurements[-1]['count']
    total_time = last_time - first_time
    total_count_diff = last_count - first_count

    if total_time <= 0 or total_count_diff <= 0:
        return None  # Avoid invalid calculations

    # Average insertions per second and minute
    average_per_second = total_count_diff / total_time
    average_per_minute = average_per_second * 60

    # Average seconds per insertion
    average_spi = total_time / total_count_diff

    # Calculate standard deviation of seconds per insertion using per-interval data
    spi_values = []
    for i in range(1, len(measurements)):
        interval_count_diff = measurements[i]['count'] - measurements[i-1]['count']
        interval_time_diff = measurements[i]['time'] - measurements[i-1]['time']
        if interval_count_diff > 0 and interval_time_diff > 0:
            spi = interval_time_diff / interval_count_diff
            spi_values.extend([spi] * interval_count_diff)

    if len(spi_values) > 1:
        mean_spi = sum(spi_values) / len(spi_values)
        variance = sum((spi - mean_spi) ** 2 for spi in spi_values) / (len(spi_values) - 1)
        std_dev_spi = math.sqrt(variance)
    else:
        std_dev_spi = float('nan')  # Undefined for one data point

    return (average_per_second, average_per_minute, average_spi, std_dev_spi)

def main():
    measurements = deque(maxlen=MEASUREMENT_WINDOW)
    initial_count = count_items()
    prev_count = initial_count
    prev_time = time.time()
    first_change_detected = False

    while True:
        current_time = time.time()
        current_count = count_items()

        # Detect first change
        if not first_change_detected:
            if current_count != prev_count:
                first_change_detected = True
                print("First change detected. Starting monitoring.")
                # Record the initial measurement
                measurements.append({'time': prev_time, 'count': prev_count})
            else:
                time.sleep(1)
                continue

        # Record the current measurement
        measurements.append({'time': current_time, 'count': current_count})

        # Update previous values
        prev_count = current_count
        prev_time = current_time

        # Calculate and log metrics
        metrics = calculate_metrics(measurements)
        if metrics is not None:
            average_per_second, average_per_minute, average_spi, std_dev_spi = metrics

            # Prepare the standard deviation string
            std_dev_str = f"{std_dev_spi:.6f}" if not math.isnan(std_dev_spi) else "undefined"

            # Log results with high precision
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_message = (
                f"{timestamp_str} | Current Count: {current_count} | "
                f"Avg Insertions/sec: {average_per_second:.6f} | "
                f"Avg Insertions/min: {average_per_minute:.6f} | "
                f"Avg Seconds per Insertion: {average_spi:.6f} | "
                f"Std Dev (Seconds per Insertion): {std_dev_str}\n"
            )

            # Write to output file
            with open(OUTPUT_FILE, 'a') as log_file:
                log_file.write(log_message)

            # Optional: Print to console
            print(log_message, end='')

        time.sleep(1)

if __name__ == "__main__":
    main()
