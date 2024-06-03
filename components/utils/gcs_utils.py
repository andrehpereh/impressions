import os
import json
from google.cloud import storage
from google.cloud import bigquery

def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """
    Uploads a file to the Google Cloud Storage bucket.

    Args:
        bucket_name (str): The name of the GCS bucket.
        source_file_name (str): The path to the file to upload.
        destination_blob_name (str): The name of the blob in the GCS bucket.
    """
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        blob.upload_from_filename(source_file_name)

        print(f"File {source_file_name} uploaded to {destination_blob_name}.")
    except Exception as e:
        print(f"An error occurred while uploading to GCS: {e}")


def download_from_gcs(bucket_name, source_blob_name, destination_file_name=None, read_only=False):
    """
    Downloads a file from the Google Cloud Storage bucket.

    Args:
        bucket_name (str): The name of the GCS bucket.
        source_blob_name (str): The name of the blob in the GCS bucket.
        destination_file_name (str): The path to save the downloaded file (ignored if read_only is True).
        read_only (bool): If True, reads and returns the content of the blob without saving to a file.
                          If False, saves the blob to the specified destination_file_name.
    """
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        print(f"Downloading blob {os.path.join(bucket_name,source_blob_name)}...")
        print("Blob exists: ", blob.exists())
        if read_only and blob.exists():
            content = blob.download_as_text()
            if content:
                print("The data in type of content is, ", type(content))
                return content
            else:
                raise ValueError("Downloaded content is empty.")
        else:
            if destination_file_name is None:
                raise ValueError("destination_file_name must be provided when read_only is False.")
            blob.download_to_filename(destination_file_name)
            print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")
    except Exception as e:
        print(f"An error occurred while downloading from GCS: {e}")

def list_gcs_files(bucket_name):
    """
    Lists all the files in the Google Cloud Storage bucket.

    Args:
        bucket_name (str): The name of the GCS bucket.
    """
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blobs = bucket.list_blobs()

        print(f"Files in bucket {bucket_name}:")
        for blob in blobs:
            print(blob.name)
    except Exception as e:
        print(f"An error occurred while listing files in GCS: {e}")

def insert_data_to_bq(dataset_id, table_id, rows_to_insert):
    """
    Inserts data into a BigQuery table.

    Args:
        dataset_id (str): The BigQuery dataset ID.
        table_id (str): The BigQuery table ID.
        rows_to_insert (list): A list of dictionaries representing rows to insert.
    """
    try:
        client = bigquery.Client()
        table_ref = client.dataset(dataset_id).table(table_id)
        table = client.get_table(table_ref)

        errors = client.insert_rows_json(table, rows_to_insert)
        if errors == []:
            print(f"Data inserted successfully into {dataset_id}.{table_id}.")
        else:
            print(f"Encountered errors while inserting rows: {errors}")
    except Exception as e:
        print(f"An error occurred while inserting data to BigQuery: {e}")

def query_bq(query):
    """
    Executes a query in BigQuery and returns the result.

    Args:
        query (str): The SQL query to execute.
    """
    try:
        client = bigquery.Client()
        query_job = client.query(query)
        results = query_job.result()

        return [dict(row) for row in results]
    except Exception as e:
        print(f"An error occurred while querying BigQuery: {e}")
        return None

def load_json_to_bq(dataset_id, table_id, json_file_path):
    """
    Loads data from a JSON file into a BigQuery table.

    Args:
        dataset_id (str): The BigQuery dataset ID.
        table_id (str): The BigQuery table ID.
        json_file_path (str): The path to the JSON file to load.
    """
    try:
        client = bigquery.Client()
        table_ref = client.dataset(dataset_id).table(table_id)

        with open(json_file_path, 'rb') as source_file:
            job_config = bigquery.LoadJobConfig(
                source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON
            )
            load_job = client.load_table_from_file(
                source_file, table_ref, job_config=job_config
            )

        load_job.result()  # Waits for the job to complete

        print(f"Loaded {load_job.output_rows} rows into {dataset_id}:{table_id}.")
    except Exception as e:
        print(f"An error occurred while loading JSON to BigQuery: {e}")

def save_dict_to_gcs(data, bucket_name, blob_name):
    """
    Saves data to a Google Cloud Storage bucket.

    Args:
        data (dict): The data to save.
        bucket_name (str): The name of the GCS bucket.
        blob_name (str): The name of the blob in the GCS bucket.
    """
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        json_data = json.dumps(data, indent=2)  # Pretty-print JSON for readability
        blob.upload_from_string(json_data, content_type='application/json')
        print(f"Data successfully saved to gs://{bucket_name}/{blob_name}")
    except Exception as e:
        print(f"An error occurred while saving to GCS: {e}")

def blob_exists(bucket_name, blob_name):
    """
    Check if a JSON file exists in the specified Google Cloud Storage bucket.

    :param bucket_name: The name of the GCS bucket.
    :param blob_name: The name of the blob (file) in the bucket.
    :return: True if the file exists, False otherwise.
    """
    # Initialize a client
    storage_client = storage.Client()

    # Get the bucket
    bucket = storage_client.bucket(bucket_name)

    # Get the blob (file) from the bucket
    blob = bucket.blob(blob_name)

    # Check if the blob exists
    return blob.exists()