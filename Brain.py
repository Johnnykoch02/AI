# Example Links to Use
links = [
    "https://en.wikipedia.org/wiki/Spanish%E2%80%93American_War",
    "https://www.britannica.com/event/Spanish-American-War",
    "https://www.history.com/topics/early-20th-century-us/spanish-american-war",
    "https://history.state.gov/milestones/1866-1898/spanish-american-war",
    "https://www.nps.gov/goga/learn/historyculture/spanish-american-war.html",
    "https://americanhistory.si.edu/price-of-freedom/spanish-american-war",
    "https://www.thoughtco.com/the-spanish-american-war-2360843"
]


import os
import openai
import time as t
import re

openai.api_key = os.getenv("OPENAI_API_KEY")
responses = []

# What information is relavent to your data extraction?
research_topic = 'the Spanish-American War'

file_obj = open('summarization_{}.txt'.format(research_topic.replace(" ", '_')),'w')

print('Collecting Summaries...')
for link in links:
    #Collect Response from Server
    print('*'*5+'Analyzing'+'*'*5)
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt="{}\nSummarize the link in 3-6 paragraphs. Include all relevant information about {}.".format(link, research_topic),
    temperature=1,
    max_tokens=3794,
    top_p=1,
    frequency_penalty=0.12,
    presence_penalty=0
    )
    # Keep track of the response for later,
    responses.append(response['choices'][0]['text'])
    file_obj.write(responses[-1]+ '\n'+ '-'*10 + '\n')
    
    # Don't Overload the server with requests
    t.sleep(20) 
    
file_obj.close()
print('Done Collecting summarizations...')
# What are you writing the essay about?
writing_prompt = 'What are the main takeaways from the Spanish-American War, and how did the United States involvement in the Spanish-American War lead to them becoming a global superpower?'

responses_string = ''
for i in responses:
    responses_string += i
responses_string = responses_string.replace('\n','').replace('  ', '')


file_obj = open('writing_prompt_{}.txt'.format(research_topic.replace(" ", '_').replace('-', '')),'w')
print('Writing Prompt...')
response = openai.Completion.create(
        model="text-davinci-003",
        prompt="{}\nWith that stated, {}.".format(responses_string, writing_prompt),
        temperature=0.9,
        max_tokens=3794,
        top_p=1,
        frequency_penalty=0.06,
        presence_penalty=0
        )   
file_obj.write(response['choices'][0]['text']+'\n')
file_obj.close()

print('You\'re Welcome, ~GPT-3 :)')
exit()