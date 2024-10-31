import time
from datetime import datetime
import math
import psycopg2

# Configuration
DB_CONFIG = {
    'host': "10.34.64.130",
    'port': "5432",
    'dbname': "postgres",
    'user': "postgres",
    'password': "admin"
}
SCHEMA_NAME = "speedbenchmark"
TABLE_NAME = "descriptor_clip"
OUTPUT_FILE = "ve_benchmark_log.txt"

def count_rows(conn):
    """Counts the number of rows in the specified database table."""
    try:
        with conn.cursor() as cur:
            query = f"SELECT COUNT(*) FROM {SCHEMA_NAME}.{TABLE_NAME}"
            cur.execute(query)
            result = cur.fetchone()
            return result[0] if result else 0
    except Exception as e:
        print(f"Error executing query: {e}")
        return None  # Return None to indicate an error

def main():
    insertion_times = []
    conn = None

    # Try to establish a connection to the database
    while conn is None:
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            conn.autocommit = True
            print("Database connection established.")
        except Exception as e:
            print(f"Error connecting to database: {e}")
            print("Retrying in 5 seconds...")
            time.sleep(5)

    initial_count = count_rows(conn)
    if initial_count is None:
        print("Error reading initial count. Will retry.")
        initial_count = 0

    prev_count = initial_count
    prev_time = time.time()
    first_change_detected = False

    try:
        while True:
            current_time = time.time()
            current_count = count_rows(conn)
            if current_count is None:
                print("Error reading current count. Retrying in 5 seconds...")
                time.sleep(5)
                continue

            # Detect first change
            if not first_change_detected:
                if current_count != prev_count:
                    first_change_detected = True
                    print("First change detected. Starting monitoring.")
                    # Record insertion times for initial difference
                    insertions = current_count - prev_count
                    insertion_times.extend([current_time] * insertions)
                else:
                    time.sleep(1)
                    continue

            # Check for new insertions
            insertions = current_count - prev_count
            if insertions > 0:
                insertion_times.extend([current_time] * insertions)

            # Update previous values
            prev_count = current_count
            prev_time = current_time

            # Calculate and log metrics if we have at least two insertions
            if len(insertion_times) >= 2:
                # Calculate inter-insertion intervals
                inter_insertion_times = [
                    t2 - t1 for t1, t2 in zip(insertion_times[:-1], insertion_times[1:])
                ]

                total_time = insertion_times[-1] - insertion_times[0]
                total_insertions = len(insertion_times) - 1  # Number of intervals

                if total_time > 0 and total_insertions > 0:
                    average_per_second = total_insertions / total_time
                    average_per_minute = average_per_second * 60
                    average_spi = total_time / total_insertions

                    # Calculate standard deviation
                    mean_spi = sum(inter_insertion_times) / total_insertions
                    variance = sum((iti - mean_spi) ** 2 for iti in inter_insertion_times) / (total_insertions - 1)
                    std_dev_spi = math.sqrt(variance)

                    # Log results
                    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_message = (
                        f"{timestamp_str} | Current Row Count: {current_count} | "
                        f"Avg Insertions/sec: {average_per_second:.6f} | "
                        f"Avg Insertions/min: {average_per_minute:.6f} | "
                        f"Avg Seconds per Insertion: {average_spi:.6f} | "
                        f"Std Dev (Seconds per Insertion): {std_dev_spi:.6f}\n"
                    )

                    # Write to output file
                    with open(OUTPUT_FILE, 'a') as log_file:
                        log_file.write(log_message)

                    # Optional: Print to console
                    print(log_message, end='')

            time.sleep(1)
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    main()
