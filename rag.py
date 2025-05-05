"""Tool module for doing RAG from a pdf file"""

# Copyright (C) 2023-2025 Cognizant Digital Business, Evolutionary AI.
# All Rights Reserved.
# Issued under the Academic Public License.
#
# You can be released from the terms, and requirements of the Academic Public
# License by purchasing a commercial license.
# Purchase of a commercial license is mandatory for any use of the
# neuro-san SDK Software in commercial settings.
#
# END COPYRIGHT

import os
import logging
from typing import Any
from typing import Dict
from .cdbvecstore import FinVecStore
from chromadb import PersistentClient
from chromadb.config import Settings

from neuro_san.interfaces.coded_tool import CodedTool

# Get absolute path to `docs` folder relative to this script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(CURRENT_DIR, "db")

class RAG(CodedTool):
    """
    CodedTool implementation which provides a way to do RAG on a pdf file
    """

    def invoke(self, args: Dict[str, Any], sly_data: Dict[str, Any]):
        """
        :param args: An argument dictionary whose keys are the parameters
            to the coded tool and whose values are the values passed for
            them by the calling agent.  This dictionary is to be treated as
            read-only.

        :param sly_data: A dictionary whose keys are defined by the agent
            hierarchy, but whose values are meant to be kept out of the
            chat stream.

            This dictionary is largely to be treated as read-only.
            It is possible to add key/value pairs to this dict that do not
            yet exist as a bulletin board, as long as the responsibility
            for which coded_tool publishes new entries is well understood
            by the agent chain implementation and the coded_tool
            implementation adding the data is not invoke()-ed more than
            once.
        """

    async def async_invoke(self, args: Dict[str, Any], sly_data: Dict[str, Any]) -> str:
        """
        Load a PDF from URL, build a vector store, and run a query against it.

        :param args: Dictionary containing 'query' (search string)
        :param sly_data: A dictionary whose keys are defined by the agent
            hierarchy, but whose values are meant to be kept out of the
            chat stream.

            This dictionary is largely to be treated as read-only.
            It is possible to add key/value pairs to this dict that do not
            yet exist as a bulletin board, as long as the responsibility
            for which coded_tool publishes new entries is well understood
            by the agent chain implementation and the coded_tool implementation
            adding the data is not invoke()-ed more than once.

            Keys expected for this implementation are:
                None
        :return: Text result from querying the built vector store,
            or error message
        """
        # Extract arguments from the input dictionary
        logging.getLogger(self.__class__.__name__)

        query: str = args.get("query", "")
        client = PersistentClient(path=DB_PATH,
                                  settings=Settings(anonymized_telemetry=False))
        collection_name = "abfrl-pdfs"
        logging.info("Connected to client at: %s", DB_PATH)
        # Validate presence of required inputs
        if not query:
            return "Error: No query provided."
        
        res = await self.query_vectorstore(client, collection_name, query)
        return res

    async def query_vectorstore(self, client, collection_name, query: str) -> str:
        """
        Query the given vector store using the provided query string
        and return the combined content of retrieved documents.

        :param vectorstore: The in-memory vector store to query
        :param query: The user query to search for relevant documents
        :return: Concatenated text content of the retrieved documents
        """
        logging.info("="*10 + "Fetching results" + "="*10)
        results = FinVecStore.query_existing_vectorstore(client, query, collection_name, top_k=10)

        logging.info("="*10 + "Completed fetching results" + "="*10)

        # Concatenate the content of all retrieved documents
        return "\n\n".join(results)
