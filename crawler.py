import warnings
import coinmarketcap as marketcap
import cointelegraph as telegraph
from threading import Thread
from queue import Queue
import csv

OUTPUT_DIR = "articles.csv"

# consume articles from the queue and write to the file
def file_writer(queue:Queue, writer:csv.DictWriter):
    while True:
        # retrieve a messages from the queue
        article = queue.get()
        # write the message to file
        writer.writerow(article)
        # mark this ask as done
        queue.task_done()

if __name__ == "__main__":
    warnings.filterwarnings("ignore", message="Unverified HTTPS request")
    # Open the CSV file for writing
    with open(OUTPUT_DIR, mode='a', newline='', encoding='utf-8') as file:
        fieldnames = ['title', 'url', 'date', 'content']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        queue = Queue()
        # Threads
        write = Thread(target=file_writer, args=(queue, writer,))
        threads = [
            Thread(target=telegraph.crawl, args=(queue,)),
            Thread(target=marketcap.crawl, args=(5,queue,)),
        ]

        write.start()
        for t in threads:
            t.start()

        for t in threads:
            t.join()
        queue.join()
        write.join()

    print('Done.')