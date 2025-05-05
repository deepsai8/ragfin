import requests
import json
import re
from bs4 import BeautifulSoup

try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


class FinDataExtractor:
    """
    A class to extract financial data from Screener.in company pages
    and format it as structured JSON suitable for vector database ingestion.
    """

    def __init__(self, url: str = None, html: str = None, use_browser: bool = False):
        """
        Initialize the extractor with either a URL or raw HTML content.

        Args:
            url (str): URL of the Screener.in company page.
            html (str): Raw HTML string (used for offline or pre-downloaded pages).
            use_browser (bool): Whether to use Playwright for dynamic content loading.
        """
        self.url = url
        self.html = html
        self.use_browser = use_browser
        self.soup = None
        self.data = {}

    def fetch_html(self):
        """Fetch the HTML content from the URL, optionally using Playwright."""
        if self.url:
            if self.use_browser and PLAYWRIGHT_AVAILABLE:
                try:
                    with sync_playwright() as p:
                        browser = p.chromium.launch(headless=True)
                        context = browser.new_context(user_agent="Mozilla/5.0")
                        page = context.new_page()
                        page.goto(self.url, wait_until="networkidle")
                        self.html = page.content()
                        browser.close()
                except Exception as e:
                    print(f"[!] Playwright failed: {e}. Falling back to requests.")
                    self._fetch_html_requests()
            else:
                self._fetch_html_requests()

    def _fetch_html_requests(self):
        """Fallback method using requests."""
        headers = {"User-Agent": "Mozilla/5.0"}
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
        return [el.get_text(strip=True, separator=" ") for el in self.soup.select(selector)]

    def extract_company_name(self):
        title = self.soup.select_one("h1")
        if title:
            self.data["company_name"] = title.text.strip()

    def extract_company_info(self):
        """
        Extracts company overview including:
        - About text
        - Key points / commentary
        - Website, BSE, NSE links
        - Key financial ratios
        """
        info = {}
        company_info = self.soup.select_one(".company-info")
        if not company_info:
            self.data["company_info"] = info
            return

        # 1. About section
        about = company_info.select_one(".company-profile .about")
        if about:
            info["about"] = about.get_text(strip=True, separator=" ")

        # 2. Key Points
        key_points = company_info.select_one(".company-profile .commentary")
        if key_points:
            info["key_points"] = key_points.get_text(separator="\n", strip=True)

        # 3. External Links (Website, BSE, NSE)
        links_block = company_info.select_one(".company-links")
        external_links = {}
        if links_block:
            for a in links_block.find_all("a"):
                label = a.get_text(strip=True, separator=" ")
                href = a.get("href")
                if label and href:
                    external_links[label] = href
        if external_links:
            info["external_links"] = external_links

        # 4. Top Ratios
        ratios = {}
        for li in company_info.select("#top-ratios li"):
            key = li.select_one(".name")
            val = li.select_one(".value")
            if key and val:
                ratios[key.text.strip()] = val.get_text(strip=True, separator=" ")
        if ratios:
            info["top_ratios"] = ratios

        self.data["company_info"] = info


    def extract_pros_cons(self):
        pros = self.extract_text_list(".pros li")
        cons = self.extract_text_list(".cons li")
        self.data["pros"] = pros
        self.data["cons"] = cons

    def extract_tables(self):
        if self.soup:
            tables = self.soup.find_all("table", class_="data-table")
            for table in tables:
                header = table.find_previous(["h2", "h3"])
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
        Includes full peer table if Playwright is enabled and rendered.
        """
        peer_section = self.soup.find("section", id="peers")
        if not peer_section:
            return

        peer_data = {}

        sub_text = peer_section.find("p", class_="sub")
        if sub_text:
            links = sub_text.find_all("a")
            if len(links) >= 1:
                peer_data["sector"] = links[0].text.strip()
            if len(links) >= 2:
                peer_data["industry"] = links[1].text.strip()

        # Benchmarks
        benchmark_tags = peer_section.select("#benchmarks a.tag")
        peer_data["benchmarks"] = [tag.text.strip() for tag in benchmark_tags]

        # Peer Table (dynamic)
        table = peer_section.find("table", class_="data-table")
        if table:
            headers = [th.text.strip() for th in table.find_all("th")]
            rows = []
            for tr in table.find_all("tr")[1:]:
                cells = [td.text.strip() for td in tr.find_all("td")]
                if cells:
                    rows.append(dict(zip(headers, cells)))
            peer_data["comparison_table"] = rows
        else:
            peer_data["comparison_table"] = "Not available in static HTML"

        self.data["peer_info"] = peer_data

    def extract_shareholding_pattern(self):
        header = self.soup.find("h2", string=re.compile("Shareholding Pattern", re.I))
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
        """
        Extract all document links including:
        - Announcements
        - Annual Reports
        - Concalls
        """
        documents = {
            "announcements": [],
            "annual_reports": [],
            "concalls": []
        }

        # --- General Documents ---
        doc_section = self.soup.find("h2", string="Documents")
        if doc_section:
            parent = doc_section.find_next("ul")
            if parent:
                for li in parent.find_all("li"):
                    a_tag = li.find("a")
                    if a_tag:
                        documents["announcements"].append({
                            "title": a_tag.text.strip(),
                            "url": a_tag.get("href")
                        })

        # --- Annual Reports ---
        ar_section = self.soup.select_one("div.documents.annual-reports ul.list-links")
        if ar_section:
            for li in ar_section.find_all("li"):
                a_tag = li.find("a")
                if a_tag:
                    title = a_tag.contents[0].strip()  # first text node
                    url = a_tag.get("href")
                    documents["annual_reports"].append({"title": title, "url": url})

        # --- Concalls ---
        cc_section = self.soup.select("div.documents.concalls ul.list-links > li")
        for li in cc_section:
            date_div = li.find("div")
            date = date_div.text.strip() if date_div else "Unknown Date"
            entries = []

            for link in li.find_all("a", class_="concall-link"):
                label = link.get_text(strip=True, separator=" ")
                url = link.get("href")
                if label and url:
                    entries.append({"type": label, "url": url})

            for button in li.find_all("button", class_="concall-link"):
                label = button.get_text(strip=True, separator=" ")
                modal_url = button.get("data-url")
                full_url = f"https://www.screener.in{modal_url}" if modal_url else None
                if label and full_url:
                    entries.append({"type": label, "url": full_url})

            if entries:
                documents["concalls"].append({"date": date, "links": entries})

        self.data["documents"] = documents

    def extract_all(self):
        """Main function to trigger parsing and extraction of all components."""
        self.parse_html()
        self.extract_company_name()
        self.extract_company_info()
        self.extract_pros_cons()
        # self.extract_peer_info()
        self.extract_tables()
        self.extract_shareholding_pattern()
        self.extract_documents_links()
        return self.data

    def to_json(self, pretty: bool = True):
        """Return the extracted data as JSON string."""
        return json.dumps(self.data, indent=2 if pretty else None)
    
    def save_json(self, filepath: str = "fin_data.json", pretty: bool = True):
        """
        Save the extracted data to a local JSON file.

        :param filepath (str): Path to the output file.
        :param pretty (bool): Whether to save with indentation and formatting.
        """
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2 if pretty else None, ensure_ascii=False)
            print(f"[+] Data saved to {filepath}")
        except Exception as e:
            print(f"[!] Failed to save JSON: {e}")


if __name__ == "__main__":
    url = "https://www.screener.in/company/AFFLE/"
    extractor = FinDataExtractor(url=url, use_browser=True)
    data = extractor.extract_all()
    # print(extractor.to_json())
    extractor.save_json("affle_fin_data.json")
