from setuptools import setup, find_namespace_packages
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), "r") as f:
    long_description = '\n' + f.read()

with open(path.join(here, 'pydparser/requirements.txt'), "r") as f:
    install_requires = f.read().splitlines()

VERSION = '1.0.4'
DESCRIPTION = 'A simple resume and job description parser used for extracting information from resumes and job descriptions and compatible with python 3.10 upwords'
setup(
    name='pydparser',
    version=VERSION,
    description=DESCRIPTION,
    url='https://github.com/justicea83/pydparser',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='justicea83',
    license='MIT',
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent"
    ],
    keywords=['python', 'resume', 'jd', 'job description', 'parser'],
    include_package_data=True,
    packages=find_namespace_packages(),
    package_data={
        "pydparser.data.jd": ['*.spacy'],
        "pydparser.data.resumes": ['*.spacy'],

        # add the models files
        "pydparser.models.jd_model": ['*.cfg', '*.json', 'tokenizer'],
        "pydparser.models.res_model": ['*.cfg', '*.json', 'tokenizer'],

        # add jd model files
        "pydparser.models.jd_model.tagger": ['cfg', 'model'],
        "pydparser.models.jd_model.tok2vec": ['cfg', 'model'],
        "pydparser.models.jd_model.vocab": ['key2row', 'lookups.bins', 'strings.json', 'vectors', 'vectors.cfg'],

        # add res model files
        "pydparser.models.res_model.ner": ['cfg', 'model', 'moves'],
        "pydparser.models.res_model.tok2vec": ['cfg', 'model'],
        "pydparser.models.res_model.vocab": ['key2row', 'lookups.bins', 'strings.json', 'vectors', 'vectors.cfg'],
    },
    install_requires=install_requires,
    zip_safe=False,
    python_requires='>=3.10'
)
