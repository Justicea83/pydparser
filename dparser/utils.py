import io
import os
import re
from collections import defaultdict

import nltk
import pandas as pd
import docx2txt
from datetime import datetime
from dateutil import relativedelta
from . import constants as cs
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFSyntaxError
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


def extract_text_from_pdf(pdf_path):
    """
    Helper function to extract the plain text from .pdf files

    :param pdf_path: path to PDF file to be extracted (remote or local)
    :return: iterator of string of extracted text
    """
    # https://www.blog.pythonlibrary.org/2018/05/03/exporting-data-from-pdfs-with-python/
    if not isinstance(pdf_path, io.BytesIO):
        # extract text from local pdf file
        with open(pdf_path, 'rb') as fh:
            try:
                for page in PDFPage.get_pages(
                        fh,
                        caching=True,
                        check_extractable=True
                ):
                    resource_manager = PDFResourceManager()
                    fake_file_handle = io.StringIO()
                    converter = TextConverter(
                        resource_manager,
                        fake_file_handle,
                        # codec='utf-8',
                        laparams=LAParams()
                    )
                    page_interpreter = PDFPageInterpreter(
                        resource_manager,
                        converter
                    )
                    page_interpreter.process_page(page)

                    text = fake_file_handle.getvalue()
                    yield text

                    # close open handles
                    converter.close()
                    fake_file_handle.close()
            except PDFSyntaxError:
                return
    else:
        # extract text from remote pdf file
        try:
            for page in PDFPage.get_pages(
                    pdf_path,
                    caching=True,
                    check_extractable=True
            ):
                resource_manager = PDFResourceManager()
                fake_file_handle = io.StringIO()
                converter = TextConverter(
                    resource_manager,
                    fake_file_handle,
                    # codec='utf-8',
                    laparams=LAParams()
                )
                page_interpreter = PDFPageInterpreter(
                    resource_manager,
                    converter
                )
                page_interpreter.process_page(page)

                text = fake_file_handle.getvalue()
                yield text

                # close open handles
                converter.close()
                fake_file_handle.close()
        except PDFSyntaxError:
            return


def get_number_of_pages(file_name):
    try:
        if isinstance(file_name, io.BytesIO):
            # for remote pdf file
            count = 0
            for page in PDFPage.get_pages(
                    file_name,
                    caching=True,
                    check_extractable=True
            ):
                count += 1
            return count
        else:
            # for local pdf file
            if file_name.endswith('.pdf'):
                count = 0
                with open(file_name, 'rb') as fh:
                    for page in PDFPage.get_pages(
                            fh,
                            caching=True,
                            check_extractable=True
                    ):
                        count += 1
                return count
            else:
                return None
    except PDFSyntaxError:
        return None


def extract_text_from_docx(doc_path):
    """
    Helper function to extract plain text from .docx files

    :param doc_path: path to .docx file to be extracted
    :return: string of extracted text
    """
    try:
        temp = docx2txt.process(doc_path)
        text = [line.replace('\t', ' ') for line in temp.split('\n') if line]
        return ' '.join(text)
    except KeyError:
        return ' '


def extract_text_from_doc(doc_path):
    """
    Helper function to extract plain text from .doc files

    :param doc_path: path to .doc file to be extracted
    :return: string of extracted text
    """
    try:
        try:
            import textract
        except ImportError:
            return ' '
        text = textract.process(doc_path).decode('utf-8')
        return text
    except KeyError:
        return ' '


def extract_text(file_path, extension):
    """
    Wrapper function to detect the file extension and call text
    extraction function accordingly

    :param file_path: path of file of which text is to be extracted
    :param extension: extension of file `file_name`
    """
    text = ''
    if extension == '.pdf':
        for page in extract_text_from_pdf(file_path):
            text += ' ' + page
    elif extension == '.docx':
        text = extract_text_from_docx(file_path)
    elif extension == '.doc':
        text = extract_text_from_doc(file_path)
    return text


def extract_entity_sections_grad(text):
    """
    Helper function to extract all the raw text from sections of
    resumes specifically for graduates and undergraduates

    :param text: Raw text of resumes
    :return: dictionary of entities
    """
    text_split = [i.strip() for i in text.split('\n')]
    # sections_in_resume = [i for i in text_split if i.lower() in sections]
    entities = {}
    key = False
    for phrase in text_split:
        if len(phrase) == 1:
            p_key = phrase
        else:
            p_key = set(phrase.lower().split()) & set(cs.RESUME_SECTIONS_GRAD)
        try:
            p_key = list(p_key)[0]
        except IndexError:
            pass
        if p_key in cs.RESUME_SECTIONS_GRAD:
            entities[p_key] = []
            key = p_key
        elif key and phrase.strip():
            entities[key].append(phrase)

    # entity_key = False
    # for entity in entities.keys():
    #     sub_entities = {}
    #     for entry in entities[entity]:
    #         if u'\u2022' not in entry:
    #             sub_entities[entry] = []
    #             entity_key = entry
    #         elif entity_key:
    #             sub_entities[entity_key].append(entry)
    #     entities[entity] = sub_entities

    # pprint.pprint(entities)

    # make entities that are not found None
    # for entity in cs.RESUME_SECTIONS:
    #     if entity not in entities.keys():
    #         entities[entity] = None
    return entities


def remove_non_readable_chars(input_string):
    """
    Remove non-human-readable characters from a string.

    This includes control characters like newline (\n), form feed (\x0c),
    and other non-printable characters.
    """

    # Regular expression for non-readable characters
    # \x00-\x1F: Control characters
    # \x7F: Delete character
    # Additional characters can be added as needed
    non_readable_regex = r'[\x00-\x1F\x7F]'

    # Replace non-readable characters with an empty string
    cleaned_string = re.sub(non_readable_regex, '', input_string)

    return cleaned_string


def is_readable(word):
    """
    Check if a word is readable.
    Strips out specific unwanted characters and then checks against a regular expression.
    """

    # Regular expression for allowed characters
    allowed_chars_regex = r'^[a-zA-Z0-9\s.,;:!?\'\"-]+$'

    return re.match(allowed_chars_regex, word) is not None


def extract_tags_with_custom_model(custom_nlp_text):
    entities = defaultdict(list)
    current_entity = []
    current_tag = ""

    for token in custom_nlp_text:
        # Remove B- or I- prefix from the tag
        tag_type = token.tag_.split('-')[-1] if '-' in token.tag_ else token.tag_

        if is_readable(token.text):
            if token.tag_.startswith('B-'):
                if current_entity:
                    entities[current_tag].append(' '.join(current_entity))
                current_tag = tag_type
                current_entity = [token.text]
            elif token.tag_.startswith('I-'):
                if not current_entity:  # Start a new entity if we haven't encountered a B- tag yet
                    current_tag = tag_type
                if tag_type == current_tag:
                    current_entity.append(token.text)
                else:
                    if current_entity:
                        entities[current_tag].append(' '.join(current_entity))
                        current_entity = [token.text]
                    current_tag = tag_type
            else:
                if current_entity:
                    entities[current_tag].append(' '.join(current_entity))
                    current_entity = []
                current_tag = ""
        else:
            # If the token is not readable, end the current entity (if any)
            if current_entity:
                entities[current_tag].append(' '.join(current_entity))
                current_entity = []
            current_tag = ""

    if current_entity:
        entities[current_tag].append(' '.join(current_entity))

    return dict(entities)


def extract_entities_wih_custom_model(custom_nlp_text):
    """
    Helper function to extract different entities with custom
    trained model using SpaCy's NER

    :param custom_nlp_text: object of `spacy.tokens.doc.Doc`
    :return: dictionary of entities
    """
    entities = {}
    for ent in custom_nlp_text.ents:
        if ent.label_ not in entities.keys():
            entities[ent.label_] = [ent.text]
        else:
            entities[ent.label_].append(ent.text)
    for key in entities.keys():
        entities[key] = list(set(entities[key]))
    return entities


def get_total_experience(experience_list):
    """
    Wrapper function to extract total months of experience from a resumes

    :param experience_list: list of experience text extracted
    :return: total months of experience
    """
    exp_ = []
    for line in experience_list:
        experience = re.search(
            r'(?P<fmonth>\w+.\d+)\s*(\D|to)\s*(?P<smonth>\w+.\d+|present)',
            line,
            re.I
        )
        if experience:
            exp_.append(experience.groups())
    total_exp = sum(
        [get_number_of_months_from_dates(i[0], i[2]) for i in exp_]
    )
    total_experience_in_months = total_exp
    return total_experience_in_months


def get_number_of_months_from_dates(date1, date2):
    """
    Helper function to extract total months of experience from a resumes

    :param date1: Starting date
    :param date2: Ending date
    :return: months of experience from date1 to date2
    """
    if date2.lower() == 'present':
        date2 = datetime.now().strftime('%b %Y')
    try:
        if len(date1.split()[0]) > 3:
            date1 = date1.split()
            date1 = date1[0][:3] + ' ' + date1[1]
        if len(date2.split()[0]) > 3:
            date2 = date2.split()
            date2 = date2[0][:3] + ' ' + date2[1]
    except IndexError:
        return 0
    try:
        date1 = datetime.strptime(str(date1), '%b %Y')
        date2 = datetime.strptime(str(date2), '%b %Y')
        months_of_experience = relativedelta.relativedelta(date2, date1)
        months_of_experience = (months_of_experience.years
                                * 12 + months_of_experience.months)
    except ValueError:
        return 0
    return months_of_experience


def extract_entity_sections_professional(text):
    """
    Helper function to extract all the raw text from sections of
    resumes specifically for professionals

    :param text: Raw text of resumes
    :return: dictionary of entities
    """
    text_split = [i.strip() for i in text.split('\n')]
    entities = {}
    key = False
    for phrase in text_split:
        if len(phrase) == 1:
            p_key = phrase
        else:
            p_key = set(phrase.lower().split()) \
                    & set(cs.RESUME_SECTIONS_PROFESSIONAL)
        try:
            p_key = list(p_key)[0]
        except IndexError:
            pass
        if p_key in cs.RESUME_SECTIONS_PROFESSIONAL:
            entities[p_key] = []
            key = p_key
        elif key and phrase.strip():
            entities[key].append(phrase)
    return entities


def extract_email(text):
    """
    Helper function to extract email id from text

    :param text: plain text extracted from resumes file
    """
    email = re.findall(r"([^@|\s]+@[^@]+\.[^@|\s]+)", text)
    if email:
        try:
            return email[0].split()[0].strip(';')
        except IndexError:
            return None


def extract_name(nlp_text, matcher):
    """
    Helper function to extract name from spacy nlp text

    :param nlp_text: object of `spacy.tokens.doc.Doc`
    :param matcher: object of `spacy.matcher.Matcher`
    :return: string of full name
    """
    pattern = [cs.NAME_PATTERN]

    matcher.add('NAME', pattern)

    matches = matcher(nlp_text)

    for _, start, end in matches:
        span = nlp_text[start:end]
        if 'name' not in span.text.lower():
            return span.text


def extract_mobile_number(text, custom_regex=None):
    """
    Helper function to extract mobile number from text

    :param text: plain text extracted from resumes file
    :return: string of extracted mobile numbers
    """
    # Found this complicated regex on :
    # https://zapier.com/blog/extract-links-email-phone-regex/
    # mob_num_regex = r'''(?:(?:\+?([1-9]|[0-9][0-9]|
    #     [0-9][0-9][0-9])\s*(?:[.-]\s*)?)?(?:\(\s*([2-9]1[02-9]|
    #     [2-9][02-8]1|[2-9][02-8][02-9])\s*\)|([0-9][1-9]|
    #     [0-9]1[02-9]|[2-9][02-8]1|
    #     [2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?([2-9]1[02-9]|
    #     [2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?([0-9]{7})
    #     (?:\s*(?:#|x\.?|ext\.?|
    #     extension)\s*(\d+))?'''

    if not custom_regex:
        mob_num_regex = r'''(\(?\d{3}\)?[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)
                                [-\.\s]*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})'''
        phone = re.findall(re.compile(mob_num_regex), text)
    else:
        phone = re.findall(re.compile(custom_regex), text)
    if phone:
        number = ''.join(phone[0])
        return number


def clean_skills(skills):
    skillset = []
    for skill in skills:
        skill = remove_non_readable_chars(skill.strip())
        if len(skill) > 0:
            skillset.append(skill)
    return skillset


def extract_skills_from_all(all_skills, noun_chunks, skills_file=None):
    if not skills_file:
        data = pd.read_csv(
            os.path.join(os.path.dirname(__file__), 'skills.csv')
        )
    else:
        data = pd.read_csv(skills_file)
    skills = list(data.columns.values)
    skillset = []

    for skill in all_skills:
        tokens = skill.lower().split()
        for token in tokens:
            if token.lower() in skills:
                skillset.append(token)

    # check for bi-grams and tri-grams
    for token in noun_chunks:
        token = token.text.lower().strip()
        if token in skills:
            skillset.append(token)

    return [i.capitalize() for i in set([i.lower() for i in skillset])]


def extract_skills(nlp_text, noun_chunks, skills_file=None):
    """
    Helper function to extract skills from spacy nlp text

    :param nlp_text: object of `spacy.tokens.doc.Doc`
    :param noun_chunks: noun chunks extracted from nlp text
    :return: list of skills extracted
    """
    tokens = [token.text for token in nlp_text if not token.is_stop]
    if not skills_file:
        data = pd.read_csv(
            os.path.join(os.path.dirname(__file__), 'skills.csv')
        )
    else:
        data = pd.read_csv(skills_file)
    skills = list(data.columns.values)
    skillset = []
    # check for one-grams
    for token in tokens:
        if token.lower() in skills:
            skillset.append(token)

    # check for bi-grams and tri-grams
    for token in noun_chunks:
        token = token.text.lower().strip()
        if token in skills:
            skillset.append(token)
    return [i.capitalize() for i in set([i.lower() for i in skillset])]


def cleanup(token, lower=True):
    if lower:
        token = token.lower()
    return token.strip()


def extract_education(nlp_text):
    """
    Helper function to extract education from spacy nlp text

    :param nlp_text: object of `spacy.tokens.doc.Doc`
    :return: tuple of education degree and year if year if found
             else only returns education degree
    """
    edu = {}
    # Extract education degree
    try:
        for index, text in enumerate(nlp_text):
            for tex in text.split():
                tex = re.sub(r'[?|$|.|!|,]', r'', tex)
                if tex.upper() in cs.EDUCATION and tex not in cs.STOPWORDS:
                    edu[tex] = text + nlp_text[index + 1]
    except IndexError:
        pass

    # Extract year
    education = []
    for key in edu.keys():
        year = re.search(re.compile(cs.YEAR), edu[key])
        if year:
            education.append((key, ''.join(year.group(0))))
        else:
            education.append(key)
    return education


def extract_experience(resume_text):
    """
    Helper function to extract experience from resumes text

    :param resume_text: Plain resumes text
    :return: list of experience
    """
    wordnet_lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # word tokenization
    word_tokens = nltk.word_tokenize(resume_text)

    # remove stop words and lemmatize
    filtered_sentence = [
        w for w in word_tokens if w not
                                  in stop_words and wordnet_lemmatizer.lemmatize(w)
                                  not in stop_words
    ]
    sent = nltk.pos_tag(filtered_sentence)

    # parse regex
    cp = nltk.RegexpParser('P: {<NNP>+}')
    cs = cp.parse(sent)

    # for i in cs.subtrees(filter=lambda x: x.label() == 'P'):
    #     print(i)

    test = []

    for vp in list(
            cs.subtrees(filter=lambda x: x.label() == 'P')
    ):
        test.append(" ".join([
            i[0] for i in vp.leaves()
            if len(vp.leaves()) >= 2])
        )

    # Search the word 'experience' in the chunk and
    # then print out the text after it
    x = [
        x[x.lower().index('experience') + 10:]
        for i, x in enumerate(test)
        if x and 'experience' in x.lower()
    ]
    return x


def extract_linkedin(text):
    """
    Helper function to extract LinkedIn URL from text

    :param text: plain text extracted from resumes file
    """
    # Regular expression to match LinkedIn URLs
    linkedin_regex = r'(https?://[a-zA-Z0-9.-]+linkedin.com/in/[^ \s]+)'

    linkedin = re.findall(linkedin_regex, text)
    if linkedin:
        return linkedin[0]
    else:
        return None


def extract_years_of_experience(text):
    """
    Extracts years of experience required from a text.
    Tries to catch various formats and edge cases.
    """
    # Patterns to look for
    patterns = [
        r'(\d+)\s+years',  # Matches '5 years', '3 years' etc.
        r'(\d+)\s*-\s*(\d+)\s+years',  # Matches '2-3 years', '3 - 5 years' etc.
        r'minimum of\s*(\d+)',  # Matches 'minimum of 3 years' etc.
        r'at least\s*(\d+)',  # Matches 'at least 2 years' etc.
        r'(\d+)\s+to\s+(\d+)\s+years',  # Matches '3 to 5 years', '2 to 3 years' etc.
        # Add more patterns as needed
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # If pattern like '2-3 years' is found, return the range
            if len(match.groups()) > 1:
                return f"{match.group(1)} to {match.group(2)} years"
            # Otherwise, return the single number found
            return f"{match.group(1)} years"

    # If no pattern is matched
    return None
