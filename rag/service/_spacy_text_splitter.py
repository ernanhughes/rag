from typing import List

import spacy


class SpacyTextSplitter:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def split_text(self, text: str) -> List[str]:
        document = self.nlp(text)
        return list(self.paragraphs(document))

    def paragraphs(self, document: str):
        start = 0
        for token in document:
            if token.is_space and token.text.count("\n") > 1:
                yield document[start:token.i]
                start = token.i
        yield document[start:]
