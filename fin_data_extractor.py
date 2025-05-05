import requests
from bs4 import BeautifulSoup
import json
import re


class FinDataExtractor:
    """
    A class to extract financial data from Screener.in company pages
    and format it as structured JSON suitable for vector database ingestion.
    """

    def __init__(self, url: str = None, html: str = None):
        """
        Initialize the extractor with either a URL or raw HTML content.

        Args:
            url (str): URL of the Screener.in company page.
            html (str): Raw HTML string (used for offline or pre-downloaded pages).
        """
        self.url = url
        self.html = html
        self.soup = None
        self.data = {}

    def fetch_html(self):
        """Fetch the HTML content from the URL."""
        if self.url:
            headers = {
                "User-Agent": "Mozilla/5.0"
                }
            response = requests.get(self.url, headers=headers)
            response.raise_for_status()
            self.html = response.text

    def parse_html(self):
        """Parse HTML using BeautifulSoup."""
        if not self.html:
            self.fetch_html()
        self.soup = BeautifulSoup(self.html, "html.parser")

    def extract_text_list(self, selector):
        """Extract list of text items using a CSS selector."""
        return [el.get_text(strip=True) for el in self.soup.select(selector)]

    def extract_company_name(self):
        title = self.soup.select_one("h1").text.strip()
        self.data["company_name"] = title

    def extract_info_box(self):
        info = {}
        for li in self.soup.select("ul.info-list li"):
            parts = li.get_text(separator=":", strip=True).split(":")
            if len(parts) == 2:
                key, value = parts
                info[key.strip()] = value.strip()
        self.data["company_info"] = info

    def extract_pros_cons(self):
        pros = self.extract_text_list(".pros li")
        cons = self.extract_text_list(".cons li")
        self.data["pros"] = pros
        self.data["cons"] = cons

    def extract_tables(self):
        """
        Extracts financial tables such as 'Quarterly Results', 'Profit & Loss', 'Balance Sheet', etc.
        """
        if self.soup:
            tables = self.soup.find_all("table", class_="data-table")
            for table in tables:
                # Identify the table by its preceding header
                header = table.find_previous("h2")
                if header:
                    table_name = header.text.strip()
                    headers = [th.text.strip() for th in table.find_all("th")]
                    rows = []
                    for tr in table.find_all("tr")[1:]:
                        cells = [td.text.strip() for td in tr.find_all("td")]
                        if cells:
                            row = dict(zip(headers, cells))
                            rows.append(row)
                    self.data[table_name] = rows

    def extract_peer_info(self):
        """
        Extract sector, industry, and benchmark index tags from the peer comparison section.
        Note: The actual peer comparison table is loaded dynamically via JS and isn't in the raw HTML.
        """
        peer_section = self.soup.find("section", id="peers")
        if not peer_section:
            return

        peer_data = {}

        # Sector and Industry
        sub_text = peer_section.find("p", class_="sub")
        if sub_text:
            links = sub_text.find_all("a")
            if len(links) >= 1:
                peer_data["sector"] = links[0].text.strip()
            if len(links) >= 2:
                peer_data["industry"] = links[1].text.strip()

        # Benchmark Indices
        benchmark_tags = peer_section.select("#benchmarks a.tag")
        peer_data["benchmarks"] = [tag.text.strip() for tag in benchmark_tags]

        self.data["peer_info"] = peer_data

    def extract_shareholding_pattern(self):
        """
        Extracts the 'Shareholding Pattern' section.
        """
        if self.soup:
            header = self.soup.find("h2", text=re.compile("Shareholding Pattern", re.I))
            if header:
                table = header.find_next_sibling("table")
                if table:
                    headers = [th.text.strip() for th in table.find_all("th")]
                    rows = []
                    for tr in table.find_all("tr")[1:]:
                        cells = [td.text.strip() for td in tr.find_all("td")]
                        if cells:
                            row = dict(zip(headers, cells))
                            rows.append(row)
                    self.data["Shareholding Pattern"] = rows

    def extract_documents_links(self):
        links = []
        doc_section = self.soup.find("h2", string="Documents")
        if doc_section:
            parent = doc_section.find_next("ul")
            if parent:
                for li in parent.find_all("li"):
                    a_tag = li.find("a")
                    if a_tag:
                        links.append({"title": a_tag.text.strip(), "url": a_tag.get("href")})
        self.data["documents"] = links

    def extract_all(self):
        """Main function to trigger parsing and extraction of all components."""
        self.parse_html()
        self.extract_company_name()
        self.extract_info_box()
        self.extract_pros_cons()
        self.extract_peer_info()
        self.extract_tables()
        self.extract_shareholding_pattern()
        self.extract_documents_links()
        return self.data

    def to_json(self, pretty: bool = True):
        """Return the extracted data as JSON string."""
        if pretty:
            return json.dumps(self.data, indent=2)
        return json.dumps(self.data)


if __name__ == "__main__":
    # Example usage:
    url = "https://www.screener.in/company/AFFLE/"
    extractor = FinDataExtractor(url)
    extractor.extract_all()
    print(extractor.to_json())
