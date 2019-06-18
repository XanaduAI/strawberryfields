from urllib.parse import urljoin
import requests


class APIClient:
    ALLOWED_BASE_URLS = [
        'localhost',
    ]
    DEFAULT_BASE_URL = 'localhost/'
    CONFIGURATION_PATH = ''

    def __init__(self, *args, **kwargs):
        # TODO: Load username, password, or authentication token from
        # configuration file

        self.USE_SSL = kwargs.get('use_ssl', True)
        self.AUTHENTICATION_TOKEN = kwargs.get('authentication_token', '')
        self.HEADERS = {}

        if 'headers' in kwargs:
            self.HEADERS.update(kwargs['headers'])

        if 'base_url' in kwargs:
            base_url = kwargs['base_url']
            if base_url in self.ALLOWED_BASE_URLS:
                self.BASE_URL = base_url
            else:
                raise ValueError('base_url parameter not in allowed list')
        else:
            self.BASE_URL = self.DEFAULT_BASE_URL

    def load_configuration(self):
        raise NotImplementedError()

    def authenticate(self, username, password):
        '''
        Retrieve an authentication token from the server via username
        and password authentication.
        '''
        raise NotImplementedError()

    def set_authorization_header(self, authentication_token):
        self.headers['Authorization'] = authentication_token

    def join_path(self, path):
        return urljoin(self.BASE_URL, path)

    def get(self, path):
        return requests.get(
            url=self.join_path(path), headers=self.headers)

    def post(self, path, payload):
        return requests.post(
            url=self.join_path(path), headers=self.headers, data=payload)


class Job:
    RESOURCE_PATH = 'jobs/'
    FIELDS = {
        "id": int,
        "status": str,
        "result_url": str,
        "circuit_url": str,
        "created_at": str,
        "started_at": str,
        "finished_at": str,
        "running_time": str,
    }

    def __init__(self, client=None, id=None, *args, **kwargs):
        if client is None:
            client = APIClient()

        self.client = client

        if id is not None:
            self.get(id)

    def join_path(self, path):
        return urljoin(self.RESOURCE_PATH, path)

    def update_job(self, data):
        for key in self.FIELDS:
            setattr(self, key, self.FIELDS[key](data.get(key)))

    def get(self, job_id):
        response = self.client.get(self.join_path(str(job_id)))
        if response.status_code == requests.status_codes.OK:
            self.update_job(response.json())
        else:
            # TODO: handle errors
            raise Exception(response.status_code)

    def create(self, params):
        # TODO do basic validation
        response = self.client.post(self.RESOURCE_PATH, params)
        if response.status_code == requests.status_codes.CREATED:
            self.update_job(response.json())
        else:
            # TODO: handle errors
            raise Exception(response.status_code)
