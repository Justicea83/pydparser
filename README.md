# pydparser

```
A simple resume and job description parser used for extracting information from resumes and job descriptions.
It also allows you to compare a set of resumes with a job description. 
```

Built with ❤️ by [Justice Arthur](https://github.com/justicea83) and inspired by  [Omkar Pathak](https://github.com/OmkarPathak/pydparser)


# Features

- Extract name from resumes
- Extract email from resumes
- Extract mobile numbers from resumes
- Extract skills from resumes
- Extract total experience from resumes
- Extract college name from resumes
- Extract degree from resumes
- Extract designation from resumes
- Extract company names from resumes
- Extract skills from job descriptions
- Extract experience level from job descriptions
- Match resumes with job descriptions and see ranking for scores

# Installation

- You can install this package using

```bash
pip install pydparser
```

- For NLP operations we use spacy and nltk. Install them using below commands:

```bash
# spaCy
python -m spacy download en_core_web_sm

# nltk
python -m nltk.downloader words
python -m nltk.downloader stopwords
```


# Supported File Formats

- PDF and DOCx files are supported on all Operating Systems
- If you want to extract DOC files you can install [textract](https://textract.readthedocs.io/en/stable/installation.html) for your OS (Linux, MacOS)
- Note: You just have to install textract (and nothing else) and doc files will get parsed easily

# Usage

- Import it in your Python project

```python
from pydparser import ResumeParser
data = ResumeParser('/path/to/resume/file').get_extracted_data()
```

```python
from pydparser import JdParser
data = JdParser('/path/to/jd/file').get_extracted_data()
```

```python
from pydparser import MatchingEngine
matcher = MatchingEngine(
    '/path/to/jd/file',
    [
        '/path/to/resume1/file',
        '/path/to/resume2/file',
        '/path/to/resume3/file',
     ]
)

# simple matcher
res = matcher.simple_intersection_score()

# similarity score with tfidf vectorizer
res = matcher.cosine_similarity_with_tfidf()

# similarity score with jaccard similarity
res = matcher.jaccard_similarity_score()
```

# CLI

For running the resume extractor you can also use the `cli` provided

```bash
usage: pydparser [-h] [-f FILE] [-d DIRECTORY] [-r REMOTEFILE]
                   [-re CUSTOM_REGEX] [-sf SKILLSFILE] [-e EXPORT_FORMAT]

optional arguments:
  -h, --help            show this help message and exit
  -f FILE, --file FILE  resume file to be extracted
  -d DIRECTORY, --directory DIRECTORY
                        directory containing all the resumes to be extracted
  -r REMOTEFILE, --remotefile REMOTEFILE
                        remote path for resume file to be extracted
  -re CUSTOM_REGEX, --custom-regex CUSTOM_REGEX
                        custom regex for parsing mobile numbers
  -sf SKILLSFILE, --skillsfile SKILLSFILE
                        custom skills CSV file against which skills are
                        searched for
  -e EXPORT_FORMAT, --export-format EXPORT_FORMAT
                        the information export format (json)
```

# Notes:

- If you are running the app on windows, then you can only extract .docs and .pdf files

# Result

The module would return a list of dictionary objects with result as follows:

```
[
  {
    'college_name': ['Marathwada Mitra Mandal’s College of Engineering'],
    'company_names': None,
    'degree': ['B.E. IN COMPUTER ENGINEERING'],
    'designation': ['Manager',
                    'TECHNICAL CONTENT WRITER',
                    'DATA ENGINEER'],
    'email': 'omkarpathak27@gmail.com',
    'mobile_number': '8087996634',
    'name': 'Omkar Pathak',
    'no_of_pages': 3,
    'skills': ['Operating systems',
              'Linux',
              'Github',
              'Testing',
              'Content',
              'Automation',
              'Python',
              'Css',
              'Website',
              'Django',
              'Opencv',
              'Programming',
              'C',
              ...],
    'total_experience': 1.83
  }
]
```


```
[{'all_skills': [
                 'building a connected and superior experience for  patients',
                 'PHP',
                 'Laravel',
                 'Object - Oriented Programming',
                 'to mentor and coach',
                 'contributing to the implementation of',
                 'mobile experience',
                 'healthcare platform',
                 'features',
                 'contribute to architectural decisions',
                 'on product excellence',
                 'building robust',
                 'PHP',
                 'Laravel',
                 'humility',
                 'celebrating bold thinking',
                 'HSA',
                 ],
  'domain': ['health', 'Public Sector', 'health service', 'mental health'],
  'experience': '5 years',
  'occupation': 'Senior Software Engineer',
  'skills': ['Mobile',
             'Budget',
             'Technical',
             'Distribution',
             'Expenses',
             'Automation',
             'Api',
             'Design',
             'Health',
             'Architectures',
             'System',
             'Access',
             'Communication',
             'Programming',
             'Php',
             'Healthcare',
             'Testing',
             'Ansible',
             'Phpunit']}]
```

# References that helped me get here

- Some of the core concepts behind the algorithm have been taken from [https://github.com/divapriya/Language_Processing](https://github.com/divapriya/Language_Processing) which has been summed up in this blog [https://medium.com/@divalicious.priya/information-extraction-from-cv-acec216c3f48](https://medium.com/@divalicious.priya/information-extraction-from-cv-acec216c3f48). Thanks to Priya for sharing this concept

- [https://www.kaggle.com/nirant/hitchhiker-s-guide-to-nlp-in-spacy](https://www.kaggle.com/nirant/hitchhiker-s-guide-to-nlp-in-spacy)

- [https://www.analyticsvidhya.com/blog/2017/04/natural-language-processing-made-easy-using-spacy-%E2%80%8Bin-python/](https://www.analyticsvidhya.com/blog/2017/04/natural-language-processing-made-easy-using-spacy-%E2%80%8Bin-python/)

- **Special thanks** to dataturks for their [annotated dataset](https://dataturks.com/blog/named-entity-recognition-in-resumes.php)
- **Special thanks** to jjzha for their [tagged dataset](https://huggingface.co/datasets/jjzha/green) which was a research by

```
@inproceedings{green-etal-2022-development,
    title = "Development of a Benchmark Corpus to Support Entity Recognition in Job Descriptions",
    author = "Green, Thomas  and
      Maynard, Diana  and
      Lin, Chenghua",
    booktitle = "Proceedings of the Thirteenth Language Resources and Evaluation Conference",
    month = jun,
    year = "2022",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2022.lrec-1.128",
    pages = "1201--1208",
}

```

# Donation

If you find this useful you don't hesitate to leave a start a something small. This encourage us to spend the Christmas creating this 😄.

| PayPal | <a href="https://paypal.me/justicearthur" target="_blank"><img src="https://www.paypalobjects.com/webstatic/mktg/logo/AM_mc_vs_dc_ae.jpg" alt="Donate via PayPal!" title="Donate via PayPal!" /></a> |
|:-------------------------------------------:|:-------------------------------------------------------------:|
