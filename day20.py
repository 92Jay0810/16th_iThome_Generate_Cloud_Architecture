import requests
from bs4 import BeautifulSoup

url = 'https://diagrams.mingrammer.com/docs/nodes/gcp'
# url = 'https://diagrams.mingrammer.com/docs/nodes/aws'
# url = 'https://diagrams.mingrammer.com/docs/nodes/azure'

response = requests.get(url)

if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')
    content_blocks = soup.find_all('p')
    for block in content_blocks:
        print(block.get_text(), end="")


else:
    print(f"Failed to retrieve the page. Status code: {response.status_code}")
