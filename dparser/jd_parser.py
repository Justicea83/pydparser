import os
import multiprocessing as mp
import io
import spacy
import pprint
from spacy.matcher import Matcher
from dparser import utils


class JdParser(object):
    def __init__(self, jd, skills_file=None, custom_regex=None):
        print('Spacy model is loading...')
        nlp = spacy.load('en_core_web_sm')

        current_directory = os.path.dirname(os.path.abspath(__file__))

        custom_nlp = spacy.load(os.path.join(current_directory, 'models', 'jd_model'))
        self.__skills_file = skills_file
        self.__custom_regex = custom_regex
        self.__matcher = Matcher(nlp.vocab)
        self.__details = {
            'domain': None,
            'skills': None,
            'qualification': None
        }
        self.__jd = jd
        if not isinstance(self.__jd, io.BytesIO):
            ext = os.path.splitext(self.__jd)[1].split('.')[1]
        else:
            ext = self.__jd.name.split('.')[1]
        self.__text_raw = utils.extract_text(self.__jd, '.' + ext)
        self.__text = ' '.join(self.__text_raw.split())
        self.__nlp = nlp(self.__text)
        self.__custom_nlp = custom_nlp(self.__text_raw)
        self.__noun_chunks = list(self.__nlp.noun_chunks)
        print('here....')
        self.__get_basic_details()

    def get_extracted_data(self):
        return self.__details

    def __get_basic_details(self):
        cust_tags = utils.extract_tags_with_custom_model(self.__custom_nlp)


def jd_result_wrapper(jd):
    parser = JdParser(jd)
    return parser.get_extracted_data()


if __name__ == '__main__':
    pool = mp.Pool(mp.cpu_count())

    jds = []
    data = []
    print('jds....')

    for root, directories, filenames in os.walk('files/jd/pdf'):
        print('d', directories)
        print('f', filenames)
        for filename in filenames:
            file = os.path.join(root, filename)
            jds.append(file)
    print(jds)

    results = [
        pool.apply_async(
            jd_result_wrapper,
            args=(x,)
        ) for x in jds
    ]

    results = [p.get() for p in results]

    pprint.pprint(results)
