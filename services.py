QUESTION_JSON_FORMAT = """
  {
    "question": "put the question here?",
    "options": [
      "A. Option A",
      "B. Option B",
      "C. Option C",
      "D. Option D"
    ],
    "answer": "put the right option here (there can only be one right options)",
    "answerKey": "For example the answer key is A"
  }

  and put the json object in list please
"""

FLASHCARD_JSON_FORMAT = """
  {
    "name": "",
    "description": ""
  }
"""

def flashcard_service(topic):
    ...