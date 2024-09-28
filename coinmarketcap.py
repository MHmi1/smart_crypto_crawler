import requests
from bs4 import BeautifulSoup
import time
import dateparser
import utills as utills
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from datetime import datetime
from queue import Queue

# ------------------------------------------------------------
# crawl article
def get_coinmarketcap_article_selenium(url,waiting_time=2):
    options = Options()
    options.headless = True  # Run headlessly
    service = Service('chromedriver.exe')  # Update with your path to chromedriver

    driver = webdriver.Chrome(service=service, options=options)
    driver.get(url)

    try:
        # Scroll down to load all dynamic content
        driver.execute_script("window.scrollTo(0, 800)")
        time.sleep(waiting_time)

        # Extract content using Selenium
        soup = BeautifulSoup(driver.page_source, 'lxml')
        driver.quit()

        title_tag = soup.find('h1')
        date_span = soup.find('span', class_='sc-e4104025-5 cpMPMB')

        # Check if title and date exist
        if title_tag:
            title = title_tag.text.strip()
        else:
            print(f"Missing title in article at {url}")
            return None
        
        if date_span:
            date_text = date_span.text.strip()
            date = dateparser.parse(date_text)  # Use dateparser to parse relative time
        else:
            print(f"Missing date in article at {url}")
            return None
        
        # Extract article content correctly
        article_tag = soup.find('article', class_='sc-65e7f566-0 sc-e4104025-0 ktlJNL wuWsT')
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
        driver.quit()
        return None

# ------------------------------------------------------------
# Find all article links
def crawl(waiting_time:int, file_queue:Queue):
    base_url = "https://coinmarketcap.com"
    sub_url = base_url + "/sitemap/community-articles/"
    
    page_number = 1

    while True:
        try:
            print(f"Fetching page {page_number}")
            headers = {'User-Agent':utills.random_agent()}
            response = requests.get(f"{sub_url}?page={page_number}", headers=headers, verify=False)
            response.raise_for_status()  # Raise an exception for HTTP errors
            soup = BeautifulSoup(response.content, 'lxml')

            # Find all article links
            links_found = 0
            for link in soup.find_all('a', href=True):
                href = link['href']
                if '/community/articles/' in href:
                    links_found += 1
                    # Construct the full URL if it's a relative URL
                    article_url = href if href.startswith('http') else base_url + href
                    article_info = get_coinmarketcap_article_selenium(article_url,waiting_time)
                    if article_info:
                        if isinstance(article_info['date'], datetime):
                            article_info['date'] = article_info['date'].strftime('%Y-%m-%d %H:%M:%S')
                        file_queue.put(article_info)
                        # save_article_to_csv(article_info,output_dir)     

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
    coinmarketcap_articles = crawl(5,queue)
    while not queue.empty():
        utills.save_article_to_csv(queue.get(), "output.csv")