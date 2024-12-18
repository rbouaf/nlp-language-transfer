import itertools
import sys
import time
import threading

def spinner():
    spinner_cycle = itertools.cycle(['|', '/', '-', '\\'])
    while not stop_spinner_event.is_set():
        sys.stdout.write(f"\r{next(spinner_cycle)}")  # Write the spinner character
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write("\r")  # Clear the spinner line when done

# Task that takes some time to complete
def task():
    time.sleep(5)  # Simulate a task taking time
    print("Task completed!")

stop_spinner_event = threading.Event()  # Event to stop the spinner

# Start the spinner in a separate thread
spinner_thread = threading.Thread(target=spinner)
spinner_thread.start()

try:
    task()  # Execute the main task
finally:
    stop_spinner_event.set()  # Signal the spinner to stop
    spinner_thread.join()  # Wait for the spinner thread to finish
