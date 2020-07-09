import logging, os
from azure.storage.blob import BlobServiceClient


def connect_to_blob_service() -> BlobServiceClient:
    """
    Generate a BlobServiceClient to connect to the Blob storage.
    """
    connect_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    logging.info(f"Connection string found: {not (connect_str is None)}")
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    return blob_service_client

def store_blobs(storage_dir, paths):
    """
    Store the payload in the Blob Storage.
    """
    #for every specified folder, for every file contained
    for relative_path in paths:
        for filename in os.listdir('./outputs/' + relative_path):
            print(f"Storing {filename} in the {storage_dir + '/' + relative_path + '/' + filename} blob")
            with open('./outputs/' + relative_path + '/' + filename, "rb") as data:
                connect_to_blob_service().get_blob_client(
                    os.getenv("DATA_CONTAINER_NAME"), storage_dir + '/' + relative_path + '/' + filename).upload_blob(data)

def remove_files_in_experiment(paths):
    # delete files that has been copied to the blob storage to save space
    for relative_path in paths:
        for filename in os.listdir('./outputs/' + relative_path):
            print(f"Deleting {relative_path + '/' + filename} from the experiment space")
            os.remove('./outputs/' + relative_path + '/' + filename)