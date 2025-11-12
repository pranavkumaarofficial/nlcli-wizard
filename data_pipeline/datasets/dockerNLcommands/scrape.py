"""
Scraper for the MattCoddity/dockerNLcommands dataset.
Downloads raw files (e.g. JSON, MD) into assets/data/base/dockerNLcommands.
"""

from scrapers.hugging_face import HuggingFaceScraper


def main():
    scraper = HuggingFaceScraper(repo="MattCoddity", dataset="dockerNLcommands")
    scraper.download_raw_files("assets/data/base/dockerNLcommands")


if __name__ == "__main__":
    main()
