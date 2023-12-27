from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from resume_parser import ResumeParser
from job_parser import JdParser
import multiprocessing as mp
import os
import pprint


def to_percentage(value):
    percentage = value * 100
    return "{:.2f}".format(percentage)


class MatchingEngine(object):
    def __init__(
            self, job_description,
            resumes: list[str],
            skills_file=None,
            custom_regex=None):
        self.jd = job_description
        self.resumes = resumes
        self.parsed_jd = JdParser(job_description).get_extracted_data()
        self.parsed_resumes = []
        for resume in self.resumes:
            self.parsed_resumes.append(ResumeParser(resume, skills_file, custom_regex).get_extracted_data())

        self.job_skills = [skill.lower() for skill in self.parsed_jd['skills']]
        self.resumes_skills = []

        for resume in self.parsed_resumes:
            batch = []
            for skill in resume['skills']:
                batch.append(skill.lower())
            self.resumes_skills.append(batch)

    def simple_intersection_score(self):
        rank = []
        for index, skills in enumerate(self.resumes_skills):
            job_skills = set(self.job_skills)
            resume_skills = set(skills)
            common_skills = set(job_skills).intersection(set(resume_skills))
            score = len(common_skills) / len(set(job_skills))
            rank.append({'name': self.parsed_resumes[index]['name'], 'score': to_percentage(score)})
        return rank

    def cosine_similarity_with_tfidf(self):
        rank = []
        for index, skills in enumerate(self.resumes_skills):
            vectorizer = TfidfVectorizer()

            vectors = vectorizer.fit_transform([" ".join(self.job_skills), " ".join(skills)])
            cosine_sim = cosine_similarity(vectors[0:1], vectors[1:2])

            score = cosine_sim[0, 0]

            rank.append({'name': self.parsed_resumes[index]['name'], 'score': to_percentage(score)})
        return rank

    def jaccard_similarity_score(self):
        rank = []
        for index, skills in enumerate(self.resumes_skills):
            job_skills = set(self.job_skills)
            resume_skills = set(skills)
            intersection = job_skills.intersection(resume_skills)
            union = job_skills.union(resume_skills)
            score = len(intersection) / len(union)
            rank.append({'name': self.parsed_resumes[index]['name'], 'score': to_percentage(score)})
        return rank


def matching_result_wrapper(jd, resumes: list[str]):
    parser = MatchingEngine(jd, resumes)
    return parser.simple_intersection_score()


if __name__ == '__main__':
    pool = mp.Pool(mp.cpu_count())

    resumes = []
    jds = []
    for root, directories, filenames in os.walk('files/res/pdf'):
        for filename in filenames:
            file = os.path.join(root, filename)
            resumes.append(file)

    for root, directories, filenames in os.walk('files/jd/pdf'):
        for filename in filenames:
            file = os.path.join(root, filename)
            jds.append(file)

    results = [
        pool.apply_async(
            matching_result_wrapper,
            args=(jds[0], resumes)
        )
    ]

    results = [p.get() for p in results]

    pprint.pprint(results)
