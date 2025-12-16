import os
import csv
import re
from pyproj import Proj
from pathlib import Path


import csv
import re
from pyproj import Proj
from pathlib import Path

def dms_to_dd(dms_tuple):
    """Convert DMS (Degrees, Minutes, Seconds) to Decimal Degrees (DD)."""
    degrees, minutes, seconds = map(float, dms_tuple)
    return degrees + minutes / 60 + seconds / 3600

def convert_to_utm(lat_dd, lon_dd, altitude):
    """Convert latitude and longitude in decimal degrees to UTM coordinates for Zone 19H (southern hemisphere)."""
    # Set up pyproj for UTM Zone 19H (southern hemisphere)
    utm_proj = Proj(proj="utm", zone=19, south=True, ellps="WGS84")
    easting, northing = utm_proj(lon_dd, lat_dd)
    return {"UTM_Easting": easting, "UTM_Northing": northing, "Altitude": altitude}

def parse_dms_string(dms_str):
    """Extract DMS tuple from string like '(36.0, 31.0, 50.3407)'."""
    match = re.match(r"\((\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)\)", dms_str)
    if match:
        return match.groups()
    return None

def process_csv_file(input_csv, output_csv):
    """Read CSV, convert coordinates, and write new CSV with UTM coordinates, retaining original columns."""
    with open(input_csv, mode='r') as infile:
        reader = csv.DictReader(infile)
        original_fieldnames = reader.fieldnames
        
        # Check if the CSV has headers; skip if it's empty or invalid
        if original_fieldnames is None:
            print(f"Skipping {input_csv}: No headers found or file is empty.")
            return

        # Add UTM columns to the original fieldnames
        fieldnames = original_fieldnames + ["UTM_Easting", "UTM_Northing", "Altitude"]

        with open(output_csv, mode='w', newline='') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                # Parse latitude and longitude in DMS format
                lat_dms = parse_dms_string(row["latitude"])
                lon_dms = parse_dms_string(row["longitude"])
                altitude = float(row["altitude"].replace(',', '.'))

                if lat_dms and lon_dms:
                    # Convert DMS to decimal degrees
                    lat_dd = dms_to_dd(lat_dms)
                    lon_dd = dms_to_dd(lon_dms)

                    # Ensure correct hemisphere for south and west coordinates
                    lat_dd = -lat_dd  # South latitude is negative
                    lon_dd = -lon_dd  # West longitude is negative

                    # Convert to UTM with corrected zone and hemisphere
                    utm_coordinates = convert_to_utm(lat_dd, lon_dd, altitude)

                    # Add UTM coordinates to the row
                    row["UTM_Easting"] = utm_coordinates["UTM_Easting"]
                    row["UTM_Northing"] = utm_coordinates["UTM_Northing"]
                    row["Altitude"] = utm_coordinates["Altitude"]

                    # Write row to new CSV
                    writer.writerow(row)

# Define other functions for managing files as in the previous code, but focusing on the processing changes here.


def delete_existing_converted_files(root_dir):
    """Delete all existing converted_ files in the directory to prevent accumulation."""
    for converted_file in Path(root_dir).rglob("converted_*.csv"):
        print(f"Deleting existing converted file: {converted_file}")
        converted_file.unlink()

def process_all_csv_files(root_dir):
    """Process all original CSV files in the inference_results directory and subdirectories."""
    root_path = Path(root_dir)
    print(f"Searching for CSV files in: {root_path}")

    # Delete existing converted files before starting
    delete_existing_converted_files(root_dir)

    csv_files = list(root_path.rglob("*.csv"))
    if not csv_files:
        print("No CSV files found.")
    else:
        for csv_file in csv_files:
            # Skip already converted files
            if "converted_" in csv_file.name:
                continue
            
            # Define the output file path with the `converted_` prefix
            output_csv_path = csv_file.with_name(f"converted_{csv_file.name}")

            # Process and save the converted file
            print(f"Processing file: {csv_file}")
            process_csv_file(csv_file, output_csv_path)
            print(f"Converted file saved as: {output_csv_path}")

# Set the root directory to the actual path of your inference_results folder
root_directory = "F:/inference_results"
process_all_csv_files(root_directory)

