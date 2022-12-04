# https://huggingface.co/

# Example Links to Use
# links = [
#     "https://en.wikipedia.org/wiki/Spanish%E2%80%93American_War",
#     "https://www.britannica.com/event/Spanish-American-War",
#     "https://www.history.com/topics/early-20th-century-us/spanish-american-war",
#     "https://history.state.gov/milestones/1866-1898/spanish-american-war",
#     "https://www.nps.gov/goga/learn/historyculture/spanish-american-war.html"
#     # "https://americanhistory.si.edu/price-of-freedom/spanish-american-war",
#     # "https://www.thoughtco.com/the-spanish-american-war-2360843"
# ]

# links = [
#     "https://en.wikipedia.org/wiki/World_War_I",
#     "https://www.britannica.com/event/World-War-I",
#     "https://www.history.com/topics/world-war-i/world-war-i-history",
#     "https://americanhistory.si.edu/topics/world-war-i",
#     "https://www.nationalgeographic.com/culture/article/world-war-i",
# ]

# links = [
#     "https://en.wikipedia.org/wiki/World_War_II",
#     "https://www.britannica.com/event/World-War-II",
#     "https://www.history.com/topics/world-war-ii/world-war-ii-history",
#     "https://americanhistory.si.edu/topics/world-war-ii",
#     "https://www.nationalgeographic.com/culture/article/world-war-ii"
# ]

links = [
    "https://en.wikipedia.org/wiki/Cold_War",
     'https://www.history.com/topics/cold-war/cold-war-history',
    "https://ehistory.osu.edu/articles/historical-analysis-cold-war",
    "https://www.timetoast.com/timelines/the-cold-war-1900-1991"  
]

import os
import openai
import time as t
import re

# You must set your API key in ur environment variables or fill it into the variable
# openai.api_key = ''
openai.api_key = os.getenv("OPENAI_API_KEY")


responses = []

# What information is relavent to your data extraction?
research_topic = 'the Cold War and Cold War Conflicts'

file_obj = open('summarization_{}.txt'.format(research_topic.replace(" ", '_')),'w')

print('Collecting Summaries...')
for link in links:
    #Collect Response from Server
    print('*'*5+'Analyzing'+'*'*5)
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt="{}\nSummarize the link in 3-6 paragraphs. Include all relevant information about {}.".format(link, research_topic),
    temperature=1,
    max_tokens=1800,
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
writing_prompt = 'What started the Cold War, what key events played a factor in US foriegn policy during the time of the Cold War, and how did it affect the US domestically?'

responses_string = ''
for i in responses:
    responses_string += i
responses_string = ''.join(responses_string.replace('\n','').replace('  ', '').split(' ')[::3000])


file_obj = open('writing_prompt_{}.txt'.format(research_topic.replace(" ", '_').replace('-', '')),'w')
print('Writing Prompt...')

response = openai.Completion.create(
        model="text-davinci-003",
        prompt="{}\nWith that stated, write a thesis about {}.".format(responses_string, writing_prompt),
        temperature=0.9,
        max_tokens=4000,
        top_p=1,
        frequency_penalty=0.06,
        presence_penalty=0
        )   

file_obj.write(response['choices'][0]['text']+'\n')

response = openai.Completion.create(
        model="text-davinci-003",
        prompt="{}\nWith that stated, write an essay about {}.".format(responses_string, writing_prompt),
        temperature=0.9,
        max_tokens=4000,
        top_p=1,
        frequency_penalty=0.06,
        presence_penalty=0
        )   

file_obj.write(response['choices'][0]['text']+'\n')

file_obj.close()

print('You\'re Welcome, ~GPT-3 :)')
exit()