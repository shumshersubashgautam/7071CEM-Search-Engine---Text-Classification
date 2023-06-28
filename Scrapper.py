
from bs4 import BeautifulSoup
import json
import requests
from typing import List


class Extract():
    '''
    This class contains tools for webscraping Coventory University website
    for School of Economics, Finance and Accounting Research

    '''
    __url: str = "https://pureportal.coventry.ac.uk/en/organisations/centre-for-intelligent-healthcare/publications/"
    __pubs: List[dict] = []
    __headers: dict = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.246"
    }
    _page_to_parse: int = 14
    __has_started: bool = False

    def __init__(self, page_to_parse: int = 14):
        '''
        Argument:
            page_to_page: Numnber of pages to parse
        '''
        self._page_to_parse = page_to_parse

    @property
    def page_to_parse(self):
        return self._page_to_parse

    @page_to_parse.setter
    def page_to_parse(self, page_to_parse):
        if not isinstance(page_to_parse, int):
            raise TypeError(f"argument 'page_to_parse' must be an integer, \
                            {type(page_to_parse)} given")

        self._page_to_parse = page_to_parse

    @property
    def pubs(self):
        if self.__has_started:
            return self.__pubs
        else:
            raise TypeError("Web scrape engine has not been executed.")

    @classmethod
    def check_author_in_author_links(cls, authors_with_link: List[str], author: str) -> bool:
        '''
        Checks if an author is represent in list of authors with links

        Arguments:
            authors_with_links: a list of authors with links
            author: Name of author to check

        Return:
            Bool
        '''

        for author_ in authors_with_link:
            if author == author_:
                return True
        return False

    def webscrape(self) -> None:
        '''
        webscrape engine
        '''
        for page in range(self._page_to_parse):
            if page != 0:
                self.__url = f"https://pureportal.coventry.ac.uk/en/organisations/centre-for-intelligent-healthcare/publications/?page={page}"

            r = requests.get(self.__url, headers=self.__headers,)
            soup = BeautifulSoup(r.content, features="lxml")

            for item in soup.find_all("li", class_="list-result-item"):
                pub_object = {
                    "publication": "",
                    "publication_link": "",
                }

                if item.find_all("a", rel="ContributionToJournal"):
                    items = item.find_all("a", rel="ContributionToJournal")
                elif item.find_all("a", rel="ContributionToBookAnthology"):
                    items = item.find_all("a", rel="ContributionToBookAnthology")

                for pub in items:
                    pub_object["publication_link"] = pub['href']
                    pub_object['publication'] = pub.find("span").text

                # get authors with link
                authors_with_link = []
                char = 'a'

                for pub in item.find_all("a", class_="link person"):
                    link = pub.get("href", None)
                    author = pub.find("span").text
                    pub_object[f"author_{char}"] = author
                    pub_object[f"author_{char}_profile"] = link

                    authors_with_link.append(f"{author}")

                    char = chr(ord(char) + 1)

                # get authors without link
                for pub in item.find_all("span", class_="")[1:-1]:
                    if not Extract.check_author_in_author_links(authors_with_link, pub.text):
                        pub_object[f"author_{char}"] = pub.text
                        pub_object[f"author_{char}_profile"] = None
                        char = chr(ord(char) + 1)

                self.__pubs.append(pub_object)

    def run(self) -> None:
        '''
        Start Web scraping
        '''
        self.webscrape()
        self.__has_started = True

    def result(self) -> None:
        if not self.__has_started:
            raise TypeError("Call start method before calling result method")

        return self.__pubs

    def result_tojson(self, path: str) -> None:
        '''
        Export results to json

        Argument:
            path: export path

        Return:
            None
        '''
        results = self.result()

        if path.split('.')[-1] != "json":
            raise ValueError('File type must be a json')

        with open(path, "w") as f:
            json.dump(results, f)


if __name__ == "__main__":
    # instantiate extraction object
    extraction = Extract()

    # run the extraction
    extraction.run()

    # export result
    extraction.result_tojson('/content/scrapped_results.json')