import hopsworks
from src.config import HOPSWORKS_PROJECT, HOPSWORKS_API_KEY

def get_project_and_fs():
    project = hopsworks.login(
        project=HOPSWORKS_PROJECT,
        api_key_value=HOPSWORKS_API_KEY
    )
    fs = project.get_feature_store()
    return project, fs