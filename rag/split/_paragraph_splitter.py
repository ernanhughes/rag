from typing import List
from ._splitter import Splitter

class ParagraphSplitter(Splitter):
    def split(self, text: str) -> List[str]:
        return text.split("\n\n")

