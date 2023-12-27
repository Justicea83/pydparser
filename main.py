from pydparser import ResumeParser

data = ResumeParser('./pydparser/files/res/pdf/test_resume.pdf').get_extracted_data()
for key, value in data.items():
    print(f"{key}\t{value}")
