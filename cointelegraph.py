from bs4 import BeautifulSoup
import time
import dateparser
import urllib.request
import utills as utills
from datetime import datetime
from queue import Queue

# ------------------------------------------------------------
# crawl article
def get_cointelegraph_article(url):
    try:
        req = urllib.request.Request(url)
        req.add_header(key='User-Agent', val=utills.random_agent())
        r = urllib.request.urlopen(req).read().decode('utf-8')
        soup = BeautifulSoup(r, 'lxml')

        # Check if title and date exist
        title_tag = soup.find('h1')
        if title_tag:
            title = title_tag.text.strip()
        else:
            print(f"Missing title in article at {url}")
            return None

        date = soup.find('time')
        if date:
            date = date.text.strip()
            date = dateparser.parse(date)
        else:
            print(f"Missing date in article at {url}")
            return None

        # Extract article content correctly
        article_tag = soup.find('div', class_='post-content relative')
        if article_tag:
            content_elements = article_tag.find_all(['div','p','h2','h3','a'])
            if content_elements:
                content = ' '.join([elem.get_text(strip=True) for elem in content_elements])
            else:
                print(f"Missing article content in article at {url}")
                return None
        else:
            print(f"Missing article content in article at {url}")
            return None

        print(f'parsed: url: {url}, date: {date}')
        return {'title': title,'url': url,'date': date,'content': content}

    except Exception as e:
        print(e)
        return None

# ------------------------------------------------------------
# Find all article links
def crawl(file_queue:Queue):
    base_url = "http://cointelegraph.com"
    sub_url = base_url + "/sitemap/post"
    page_number = 1
    while True:
        try:
            req = urllib.request.Request(f"{sub_url}-{page_number}.xml")
            req.add_header(key='User-Agent', val=utills.random_agent())
            r = urllib.request.urlopen(req).read().decode('utf-8')
            soup = BeautifulSoup(r, 'xml')
            # Find all article links
            links_found = 0
            for link in soup.find_all('loc'):
                href = link.get_text()
                if 'news' in href:
                    # Construct the full URL if it's a relative URL
                    article_url = href if href.startswith('http') else base_url + href
                    links_found += 1
                    article_info = get_cointelegraph_article(article_url)
                    if article_info:
                        # Convert the date to string format if it's a datetime object
                        if isinstance(article_info['date'], datetime):
                            article_info['date'] = article_info['date'].strftime('%Y-%m-%d %H:%M:%S')
                        file_queue.put(article_info)
                        # save_article_to_csv(article_info, output_dir)

            if links_found == 0:
                # No more articles found, break out of the loop
                break

            page_number += 1
            time.sleep(10)  # Pause between requests to avoid rate-limiting

        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(10)  # Exponential backoff
            continue

# ------------------------------------------------------------
if __name__ == "__main__":
    queue = Queue()
    cointelegraph_articles = crawl(queue)
    while not queue.empty():
        utills.save_article_to_csv(queue.get(), "output.csv")   