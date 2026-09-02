"""
Batch execution helper for parameter indexing from command line.
Usage: python run_batch.py <index>
"""
import sys

index = 0  # Default index value

# If an index argument is passed from the command line, overwrite index
if len(sys.argv) > 1:
    index = int(sys.argv[1])
    print(f"Running MCMC with dynamic index: {index}")
else:
    print(f"Running MCMC with default index: {index}")
