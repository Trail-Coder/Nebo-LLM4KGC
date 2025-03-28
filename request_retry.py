import time
import requests
def send_request_with_retries(proxy_api_url, headers, data, max_retries=10, retry_interval=1):
    for retry in range(max_retries):
        try:
            response = requests.post(proxy_api_url, headers=headers, data=data)
            if response.status_code == 200:
                return response  
            print(f"Request failed with status code {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Request failed with error: {e}")
        if retry < max_retries - 1:
            print(f"Retrying... (retry {retry + 1}/{max_retries})")
            time.sleep(retry_interval)
    print("\n----------------------\nMax retries exceeded, giving up.\n----------------------\n")
    return None 