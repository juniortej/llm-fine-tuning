import os
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
import json
from tqdm import tqdm
from cerebras.cloud.sdk import Cerebras

"""
This is a script to generate conversations based on a dataset of essays.
I then used these conversations to finetune a model that should resemble Marie Kondo in this case.

All essays:

"""

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv("config/.env.secret")

# MODEL - better to use different models for different tasks
MODEL_LLAMA = "llama-4-scout-17b-16e-instruct"
MODEL_LLAMA_3 = "llama-3.3-70b"
MODEL_LLAMA_8B = "llama3.1-8b"

client = Cerebras(
    api_key=os.environ.get("CEREBRAS_API_KEY"),
)
# ----------------- PARAMETERS -----------------

# GENERATE
GENERATE_GREETINGS = False
GENERATE_BASIC_INSTRUCTIONS = False
GENERATE_QUESTION_ANSWER_PAIRS = True
GENERATE_ADVICE_ANSWER_PAIRS = True

# MODELS
GREETINGS_MODEL = MODEL_LLAMA
QA_MODEL = MODEL_LLAMA_3
ADVICE_MODEL = MODEL_LLAMA
CHARACTER_PROFILE_MODEL = MODEL_LLAMA_8B

# MODEL PARAMETERS
temperature = 1  # Higher temperature means more randomness maximum is 2 default is 1

# TOTAL DATA
ESSAY_CHUNK_SIZE = 500
QUESTIONS_PER_CHUNK = 100
ADVICE_PER_CHUNK = 100

# COUNTS
BASIC_INSTRUCTIONS_COUNT = 50
GREETING_COUNT = 100

# PROMPTS
GREETING_FROM_USER_PROMPT = """
Generate a greeting for the start of a conversation. ONLY GIVE THE GREETING BACK.
"""

GREETING_FROM_ASSISTANT_PROMPT = """
Pretend you are Marie Kondo. Generate a short response to the greeting. ONLY GIVE THE RESPONSE BACK.

Greeting:
{greeting}
"""

CHARACTER_QUESTION_PROMPTS = [
    "Ask my name. Just give me the question alone nothing else.",
    "Ask where I was born. Just give me the question alone nothing else.",
    "Ask my age. Just give me the question alone nothing else.",
    "Ask my what I do. Just give me the question alone nothing else.",
    "Ask where I live. Just give me the question alone nothing else.",
]

CHARACTER_ANSWER_PROMPT = """
Pretend you are Marie Kondo, you were born on 9 October 1984 in Osaka, Japan. You are known for your tidying expertise and the KonMari Method. You are also a writer and essayist. You live in Japan. The current year is 2025.
Answer the following question.

QUESTION:
{question}
"""

ESSAY_QUESTION_PROMPT = """
Generate a short question for your upcoming interview with the author of this text. Your short interview question however needs to able to be answered using the info found in ESSAY TEXT. The question should be about the thought process of the author. Remember the question has to be short and ONLY GIVE THE QUESTION BACK.:"

ESSAY TEXT:
{essay_text}

QUESTION:
"""

ESSAY_QUESTION_ANSWER_PROMPT = """
Pretend you are Marie Kondo, the author of the following essay doing an interview about the essay. Answer concisely to the question of using information found in your essay. Adopt the conversation style of Marie Kondo. ONLY GIVE THE ANSWER BACK.

ESSAY TEXT: 
{essay_text} 

QUESTION:
{question_text}

ANSWER:

"""

ESSAY_ASK_ADVICE_PROMPT = """
Ask for advice about something discussed in the following essay. Format the question in the "I" form, like you have the problem yourself. ONLY GIVE THE QUESTION BACK.:"

ESSAY TEXT:
{essay_text}

QUESTION:
"""

ESSAY_ADVICE_ANSWER_PROMPT = """
Pretend you are Marie Kondo and adopt the same conversation style as her. Give advice for the QUESTION given information found in your ESSAY_TEXT. Adopt the conversation style of Marie Kondo and answer concisely (similar to how she talks in the essay). ONLY GIVE THE ANSWER BACK.

ESSAY TEXT:
{essay_text} 

QUESTION: 
{question_text}

ANSWER:
"""
# ----------------- END OF PARAMETERS -----------------


@retry(
    stop=stop_after_attempt(5),  # Stop after 5 attempts
    wait=wait_exponential(min=1, max=100),  # Wait exponentially between retries
)
def ask_ai(question, model):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": question,
            }
        ],
        model=model,
        temperature=temperature,
        max_tokens=600
    )
    return chat_completion.choices[0].message.content


def generate_question_about_essay(essay_text, model):
    prompt = ESSAY_QUESTION_PROMPT.format(essay_text=essay_text)
    return ask_ai(prompt, model=model)


def generate_answer_to_question_about_essay(essay_text, question_text, model):
    prompt = ESSAY_QUESTION_ANSWER_PROMPT.format(
        essay_text=essay_text, question_text=question_text
    )
    return ask_ai(prompt, model=model)


def ask_advice_about_essay(essay_text, model):
    prompt = ESSAY_ASK_ADVICE_PROMPT.format(essay_text=essay_text)
    return ask_ai(prompt, model=model)


def give_advice_about_essay(essay_text, question_text, model):
    prompt = ESSAY_ADVICE_ANSWER_PROMPT.format(
        essay_text=essay_text, question_text=question_text
    )
    return ask_ai(prompt, model=model)


def get_essay_text_from_file(file_path):
    with open(file_path, "r") as file:
        return file.read()


def get_file_paths_from_directory(directory):
    return [os.path.join(directory, filename) for filename in os.listdir(directory)]


# Split essay text into 1000 character chunks
def split_essay_text_into_chunks(essay_text, chunk_size=1500):
    chunks = []
    current_chunk = ""
    for word in essay_text.split():
        if len(current_chunk) < chunk_size:
            current_chunk += word + " "
        else:
            chunks.append(current_chunk)
            current_chunk = ""
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def generate_question_answer_pairs_from_essays_directory(directory):
    file_paths = get_file_paths_from_directory(directory)
    conversations = []
    for i in tqdm(range(len(file_paths))):
        file_path = file_paths[i]
        essay_text = get_essay_text_from_file(file_path)
        essay_text_chunks = split_essay_text_into_chunks(
            essay_text, chunk_size=ESSAY_CHUNK_SIZE
        )
        for chunk in essay_text_chunks:
            for _ in range(QUESTIONS_PER_CHUNK):
                question = generate_question_about_essay(chunk, model=QA_MODEL)
                answer = generate_answer_to_question_about_essay(
                    chunk, question, model=QA_MODEL
                )
                conversations.append(
                    [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer},
                    ]
                )
            break
    return conversations


def generate_advice_answer_pairs_from_essays_directory(directory):
    file_paths = get_file_paths_from_directory(directory)
    conversations = []
    for i in tqdm(range(len(file_paths))):
        file_path = file_paths[i]
        essay_text = get_essay_text_from_file(file_path)
        essay_text_chunks = split_essay_text_into_chunks(essay_text)
        for chunk in essay_text_chunks:
            question = ask_advice_about_essay(chunk, model=ADVICE_MODEL)
            answer = give_advice_about_essay(chunk, question, model=ADVICE_MODEL)
            conversations.append(
                [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ]
            )
            break
    return conversations


def generate_greetings(count=40):
    conversations = []
    for _ in tqdm(range(count)):
        question = ask_ai(GREETING_FROM_USER_PROMPT, model=GREETINGS_MODEL)
        answer_prompt = GREETING_FROM_ASSISTANT_PROMPT.format(greeting=question)
        answer = ask_ai(answer_prompt, model=GREETINGS_MODEL)
        conversations.append(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        )
    return conversations


def generate_character_basic_instructions(count=40):
    conversations = []
    for i in tqdm(range(count)):
        if i % 5 == 0:
            ask_prompt = CHARACTER_QUESTION_PROMPTS[0]
        elif i % 5 == 1:
            ask_prompt = CHARACTER_QUESTION_PROMPTS[1]
        elif i % 5 == 2:
            ask_prompt = CHARACTER_QUESTION_PROMPTS[2]
        elif i % 5 == 3:
            ask_prompt = CHARACTER_QUESTION_PROMPTS[3]
        else:
            ask_prompt = CHARACTER_QUESTION_PROMPTS[4]
        question = ask_ai(ask_prompt, model=CHARACTER_PROFILE_MODEL)
        answer_prompt = CHARACTER_ANSWER_PROMPT.format(question=question)
        answer = ask_ai(answer_prompt, model=CHARACTER_PROFILE_MODEL)
        conversations.append(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        )
    return conversations


# save conversations to jsonl file
def save_conversations_to_jsonl(conversations, outdir, file_name):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    file_path = os.path.join(outdir, file_name)
    with open(file_path, "w") as file:
        for conversation in conversations:
            item = {"conversations": conversation}
            json_string = json.dumps(item)  # Serialize to a JSON formatted string
            file.write(f"{json_string}\n")


def main():
    essays_dir = "dataset-preparation/conversation/sample-dataset/someone_essays/" #for marie kondo
    output_dir = "finetune_dataset/datasets/mk/conversation/"

    if GENERATE_GREETINGS:
        print("Generating greetings")
        greeting_conversations = generate_greetings(count=GREETING_COUNT)
        file_name = "someone_greetings.jsonl"
        save_conversations_to_jsonl(greeting_conversations, output_dir, file_name)

    if GENERATE_BASIC_INSTRUCTIONS:
        print("Generating basic instructions from essays directory")
        basic_conversations = generate_character_basic_instructions(
            count=BASIC_INSTRUCTIONS_COUNT
        )
        file_name = "someone_basic_instructions.jsonl"
        save_conversations_to_jsonl(basic_conversations, output_dir, file_name)

    if GENERATE_QUESTION_ANSWER_PAIRS:
        print("Generating question answer pairs from essays directory")
        qa_conversations = generate_question_answer_pairs_from_essays_directory(
            essays_dir
        )
        file_name = "someone_essays_qa_conversations.jsonl"
        save_conversations_to_jsonl(qa_conversations, output_dir, file_name)

    if GENERATE_ADVICE_ANSWER_PAIRS:
        print("Generating advice answer pairs from essays directory")
        advice_conversations = generate_advice_answer_pairs_from_essays_directory(
            essays_dir
        )
        file_name = "someone_essays_advice_conversations.jsonl"
        save_conversations_to_jsonl(advice_conversations, output_dir, file_name)


main()
