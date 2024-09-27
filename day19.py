import requests
from bs4 import BeautifulSoup

# url = 'https://diagrams.mingrammer.com/docs/guides/diagram'
# url = 'https://diagrams.mingrammer.com/docs/guides/node'
# url = 'https://diagrams.mingrammer.com/docs/guides/cluster'
url = 'https://diagrams.mingrammer.com/docs/guides/edge'

response = requests.get(url)

if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')
    content_blocks = soup.find_all(['p', 'pre'])
    for block in content_blocks:
        if block.name == 'p':
            print("Paragraph:", block.get_text())
        elif block.name == 'pre':
            print("Code Block:\n", block.get_text())

else:
    print(f"Failed to retrieve the page. Status code: {response.status_code}")
