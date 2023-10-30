from llama_index import SimpleDirectoryReader, GPTListIndex, readers, GPTSimpleVectorIndex, LLMPredictor, PromptHelper, ServiceContext
from langchain import OpenAI
import sys
import os
from IPython.display import Markdown, display

def construct_index(directory_path):
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 2000
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chunk size limit
    chunk_size_limit = 600

    # define prompt helper
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    # define LLM
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.5, model_name="text-davinci-003", max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)

    index.save_to_disk('index.json')

    return index


# Define a function to preprocess text
def preprocess_text(text):
    # Remove special characters
    processed_text = re.sub(r'[^\w\s]', '', text)
    return processed_text

# Define a function to generate responses
def generate_response(index, llm_predictor, prompt):
    response = index.query(prompt)
    llm_response = llm_predictor.predict(response.response)
    return llm_response

# Define a function to extract action items
def extract_action_items(llm_predictor, chat_history):
    action_items_prompt = "Try this prompt for Action Item\n\n" + \
                          "CONTENT STARTS HERE.\n" + \
                          f"{chat_history}\n" + \
                          "CONTENT STOPS HERE.\n" + \
                          "+++++"
    
    llm_response = llm_predictor.predict(action_items_prompt)
    extracted_data = json.loads(llm_response.choices[0].text)
    return extracted_data.get("actionItems", [])



def calculate_match_percentage(candidate_data, job_description):
    # Convert candidate_data and job_description to lowercase for case-insensitive comparison
    candidate_data = candidate_data.lower()
    job_description = job_description.lower()
    
    # Split the text into words
    candidate_words = set(candidate_data.split())
    job_words = set(job_description.split())
    
    # Calculate the intersection (matching words) and union (total unique words)
    common_words = candidate_words.intersection(job_words)
    total_unique_words = candidate_words.union(job_words)
    
    # Calculate the match percentage based on the ratio of common words to total unique words
    match_percentage = (len(common_words) / len(total_unique_words)) * 100
    
    return match_percentage

# Test the function
resume_data = "Experienced software engineer with a strong background in machine learning."
linkedin_data = "Skilled in data analysis and problem-solving."
github_data = "Contributed to open-source projects related to natural language processing."
job_description = "Looking for a software engineer with expertise in machine learning and data analysis."

match_percentage = calculate_match_percentage(
    resume_data + linkedin_data + github_data,
    job_description
)

print(f"Match Percentage: {match_percentage:.2f}%")


def ask_ai():
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    while True:
        # prompt_type = input("Select prompt type (action/summary/rollback): ")
        prompt_type = input("Select prompt type (action/summary/rollback/candidate_analysis): ")
        if prompt_type == "candidate_analysis":
        # Get file paths for candidate documents and job description
            resume_path = input("Upload the resume file path: ")
            linkedin_path = input("Upload the LinkedIn profile file path: ")
            github_path = input("Upload the GitHub profile file path: ")
            job_description = input("Enter the job description: ")
        
        # Load and preprocess candidate data and job description
        with open(resume_path, 'r') as file:
            resume_data = preprocess_text(file.read())
        # Repeat for LinkedIn and GitHub profiles
        
        # Calculate match percentage
        match_percentage = calculate_match_percentage(
            resume_data + linkedin_data + github_data,
            job_description
        )
        
        print(f"Match Percentage: {match_percentage}%")
    

        response = index.query(prompt)
        print("Response:", response.response)



# Make sure you have your environment variable and construct_index() call here
os.environ["OPENAI_API_KEY"] = ""
construct_index("nite")

# Define your prompts here
action_items_prompt = "Try this prompt for Action Item\n\n" + \
                     "@\"You are an action item extractor. You will be given chat history and need to make note of action items mentioned in the chat.\n" + \
                     "Extract action items from the content if there are any. If there are no action, return nothing. If a single field is missing, use an empty string.\n" + \
                     "Return the action items in json.\n\n" + \
                     "Possible statuses for action items are: Open, Closed, In Progress.\n\n" + \
                     "EXAMPLE INPUT WITH ACTION ITEMS:\n" + \
                     "John Doe said: \"I will record a demo for the new feature by Friday\"\n" + \
                     "I said: \"Great, thanks John. We may not use all of it but it's good to get it out there.\"\n\n" + \
                     "EXAMPLE OUTPUT:\n" + \
                     "{\n" + \
                     "    \"actionItems\": [\n" + \
                     "        {\n" + \
                     "            \"owner\": \"John Doe\",\n" + \
                     "            \"actionItem\": \"Record a demo for the new feature\",\n" + \
                     "            \"dueDate\": \"Friday\",\n" + \
                     "            \"status\": \"Open\",\n" + \
                     "            \"notes\": \"\"\n" + \
                     "        }\n" + \
                     "    ]\n" + \
                     "}\n\n" + \
                     "EXAMPLE INPUT WITHOUT ACTION ITEMS:\n" + \
                     "John Doe said: \"Hey I'm going to the store, do you need anything?\"\n" + \
                     "I said: \"No thanks, I'm good.\"\n\n" + \
                     "EXAMPLE OUTPUT:\n" + \
                     "{\n" + \
                     "    \"action_items\": []\n" + \
                     "}\n\n" + \
                     "CONTENT STARTS HERE.\n" + \
                     "{{$INPUT}}\n" + \
                     "CONTENT STOPS HERE.\n" + \
                     "+++++"

summary_prompt = "[SUMMARIZATION RULES]\nDONT WASTE WORDS\nUSE SHORT, CLEAR, COMPLETE SENTENCES.\n" + \
                 "DO NOT USE BULLET POINTS OR DASHES.\nUSE ACTIVE VOICE.\nMAXIMIZE DETAIL, MEANING\nFOCUS ON THE CONTENT\n\n" + \
                 "[BANNED PHRASES]\nThis article\nThis document\nThis page\nThis material\n[END LIST]\n\n" + \
                 "Summarize:\n{{$input}}\n+++++"

rollback_prompt = "Suggest potential rollbacks from the transcript:\n\n" + \
                  "Rollback:\n{{$input}}\n+++++"

ask_ai()
