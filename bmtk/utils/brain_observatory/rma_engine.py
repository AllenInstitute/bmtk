import pandas as pd
import time
import requests

from .utils import infer_column_types

try:
    from tqdm import tqdm
except ImportError:
    from .utils import FakeTqdm as tqdm



DEFAULT_TIMEOUT = 20 * 60  # seconds
DEFAULT_CHUNKSIZE = 1024 * 10  # bytes


class HttpEngine:
    def __init__(
        self, 
        scheme: str, 
        host: str, 
        timeout: float = DEFAULT_TIMEOUT, 
        chunksize: int = DEFAULT_CHUNKSIZE,
        **kwargs
    ):
        self.scheme = scheme
        self.host = host
        self.timeout = timeout
        self.chunksize = chunksize


    def stream(self, route):
        """ Makes an http request and returns an iterator over the response.

        Parameters
        ----------
        route :
            the http route (under this object's host) to request against.

        """

        url = self._build_url(route)
        
        start_time = time.perf_counter()
        response = requests.get(url, stream=True)
        response_b = None
        if "Content-length" in response.headers:
            response_b = float(response.headers["Content-length"])

        size_message = f"{response_b / 1024 ** 2:3.3f}MiB" if response_b is not None else "potentially large"
        # logging.warning(f"downloading a {size_message} file from {url}")
        progress = tqdm(unit="B", total=response_b, unit_scale=True,  desc="Downloading")

        for chunk in response.iter_content(self.chunksize):
            if chunk: # filter out keep-alive new chunks
                progress.update(len(chunk))
                yield chunk

            elapsed = time.perf_counter() - start_time
            if elapsed > self.timeout:
                raise requests.Timeout(f"Download took {elapsed} seconds, but timeout was set to {self.timeout}")

    def _build_url(self, route):
        return f"{self.scheme}://{self.host}/{route}"



class RmaEngine(HttpEngine):
    def __init__(
        self, 
        scheme, 
        host, 
        rma_prefix: str = "api/v2/data", 
        rma_format: str = "json", 
        page_size: int = 5000, 
        **kwargs
    ):
        super(RmaEngine, self).__init__(scheme, host, **kwargs)
        self.rma_prefix = rma_prefix
        self.rma_format = rma_format
        self.page_size = page_size

    @property
    def format_query_string(self):
        return f"query.{self.rma_format}"

    def add_page_params(self, url, start, count=None):
        if count is None:
            count = self.page_size
        return f"{url},rma::options[start_row$eq{start}][num_rows$eq{count}][order$eq'id']"


    def get_rma(self, query: str):
        """ Makes a paging rma query

        Parameters
        ----------
        query : 
            The RMA query parameters

        """
        url = f"{self.scheme}://{self.host}/{self.rma_prefix}/{self.format_query_string}?{query}"
        # logging.debug(url)

        start_row = 0
        total_rows = None

        start_time = time.time()
        while total_rows is None or start_row < total_rows:
            current_url = self.add_page_params(url, start_row)
            response_json = requests.get(current_url).json()
            if not response_json["success"]:
                raise Exception(response_json["msg"])

            start_row += response_json["num_rows"]
            if total_rows is None:
                total_rows = response_json["total_rows"]

            # logging.debug(f"downloaded {start_row} of {total_rows} records ({time.time() - start_time:.3f} seconds)")
            yield response_json["msg"]


    def get_rma_list(self, query):
        response = []
        for chunk in self.get_rma(query):
            response.extend(chunk)
        return response

    def get_rma_tabular(self, query, try_infer_dtypes=True):
        response = pd.DataFrame(self.get_rma_list(query))

        if try_infer_dtypes:
            response = infer_column_types(response)

        return response