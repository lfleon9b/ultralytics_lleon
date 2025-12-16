import json
import os
import pyproj
import csv

# Geo Data section

# GeoJSON generation function

import json
import os
import pyproj

def generate_geojson(image_name, metadata_path, output_dir, styled_image_path):
    # Load metadata from JSON file
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Try to extract UTM coordinates, handle missing coordinates
    utm_easting = metadata.get('UTM_Coordinates', {}).get('UTM_Easting', '')
    utm_northing = metadata.get('UTM_Coordinates', {}).get('UTM_Northing', '')
    altitude = metadata.get('UTM_Coordinates', {}).get('Altitude', '')

    if not utm_easting or not utm_northing:
        print(f"UTM coordinates are missing for {image_name}, skipping GeoJSON generation.")
        return

    try:
        zone_number = 19  # UTM zone number
        hemisphere = 'S'  # Adjust based on your location ('N' for Northern, 'S' for Southern)

        # Create CRS for UTM depending on the hemisphere
        if hemisphere == 'S':
            utm_proj = pyproj.CRS(f"+proj=utm +zone={zone_number} +south +datum=WGS84")
        else:
            utm_proj = pyproj.CRS(f"+proj=utm +zone={zone_number} +datum=WGS84")

        wgs84_proj = pyproj.CRS("EPSG:4326")  # Standard EPSG code for WGS84 lat/lon

        # Create a transformer for the conversion
        transformer = pyproj.Transformer.from_crs(utm_proj, wgs84_proj, always_xy=True)

        # Perform the transformation
        lon, lat = transformer.transform(utm_easting, utm_northing)
    except Exception as e:
        print(f"Failed to convert UTM to lat/lon for {image_name}: {str(e)}")
        return

    # Extract model_version and other model info from the nested 'model_info' key
    model_version = metadata['model_info']['model_version']
    confidence_threshold = metadata['model_info']['confidence_threshold']
    img_size = metadata['model_info']['img_size']
    training_image_size = metadata['model_info']['training_image_size']
    source_path = metadata['model_info']['source_path']

    # Create the GeoJSON structure with both lat/lon and UTM coordinates
    geojson_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "image_name": image_name,
                    "label_summary": metadata['label_summary'],
                    "model_version": model_version,
                    "confidence_threshold": confidence_threshold,
                    "img_size": img_size,
                    "training_image_size": training_image_size,
                    "source_path": source_path,
                    "styled_image_link": styled_image_path,
                    "UTM_Coordinates": {
                        "UTM_Easting": utm_easting,
                        "UTM_Northing": utm_northing,
                        "Altitude": altitude
                    }
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat]  # Use lat/lon coordinates
                }
            }
        ]
    }

    # Save the GeoJSON file
    geojson_filename = os.path.join(output_dir, f"{image_name}.geojson")
    with open(geojson_filename, 'w') as f:
        json.dump(geojson_data, f, indent=4)
    
    print(f"GeoJSON saved for {image_name}: {geojson_filename}")


def gather_unique_labels(batch_dir):
    """
    Gather all unique labels from the batch of images by scanning the Label_Summary fields
    in the metadata files.

    Args:
    batch_dir (str): The directory containing the batch of images with JSON metadata.

    Returns:
    set: A set of unique labels/classes found in all images.
    """
    unique_labels = set()

    # Traverse through the batch directory to find JSON metadata files
    for root, dirs, files in os.walk(batch_dir):
        for file in files:
            if file.endswith('_image_metadata.json'):
                json_metadata_path = os.path.join(root, file)
                with open(json_metadata_path, 'r') as f:
                    metadata = json.load(f)
                    label_summary = metadata['label_summary']
                    unique_labels.update(label_summary.keys())  # Add all keys (labels) to the set

    return unique_labels



def generate_batch_summary_csv(batch_dir, output_csv, results):
    """
    Generate a CSV file summarizing the UTM coordinates, label summary, image paths, and processing times for a batch of images.
    """
    unique_labels = set()
    for result in results:
        unique_labels.update(result['label_summary'].keys())
    unique_labels = sorted(unique_labels)

    # Define the header for the CSV file
    csv_header = ['Image_Name', 'UTM_Easting', 'UTM_Northing', 'Altitude', 'Image_Path', 'Styled_Image_Path', 'Processing_Time'] + unique_labels

    # Create the CSV file and write the header
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(csv_header)

        # Traverse through the results and write each row
        for result in results:
            image_name = result['image_name']
            utm_easting = result.get('utm_easting', '')
            utm_northing = result.get('utm_northing', '')
            altitude = result.get('altitude', '')
            image_path = result['image_path']
            styled_image_path = result['styled_image_path']
            processing_time = result['processing_time']
            label_summary = result['label_summary']

            # Initialize row data with basic info
            row_data = [image_name, utm_easting, utm_northing, altitude, image_path, styled_image_path, processing_time]

            # Add counts for each label; use 0 if the label is not present in this image's label_summary
            for label in unique_labels:
                row_data.append(label_summary.get(label, 0))

            # Write the row to the CSV file
            csv_writer.writerow(row_data)

    print(f"CSV summary generated at: {output_csv}")


