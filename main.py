import openai
from dotenv import load_dotenv
from constants import openai_key
import os
import streamlit as st
from langchain_community.llms import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
# Load environment variables
os.environ["OPENAI_API_KEY"] = openai_key

# Initialize OpenAI instance
llm = OpenAI(temperature=0.8)

# Streamlit application
st.title('Celebrity Search Results')
input_text=st.text_input("Search the topic u want")


person_memory=ConversationBufferMemory(input_key='name',memory_key="chat_history")
dob_memory=ConversationBufferMemory(input_key='name',memory_key="chat_history")
desc_memory=ConversationBufferMemory(input_key='dob',memory_key="description_history")


# Prompt Templates

first_input_prompt=PromptTemplate(
    input_variables=['name'],
    template="Tell me about celebrity {name}"
)
chain = LLMChain(llm=llm, prompt=first_input_prompt, verbose=True, output_key='person',memory=person_memory)






second_input_prompt = PromptTemplate(
    input_variables=["name"],
    template="When was {name} born. Give result in only date format(eg:07-092001)"
)
chain2 = LLMChain(llm=llm, prompt=second_input_prompt, verbose=True, output_key='dob',memory=dob_memory)


Third_input_prompt = PromptTemplate(
    input_variables=["dob"],
    template="Mention 5 major events happened around {dob} in the world"
)
chain3 = LLMChain(llm=llm, prompt=Third_input_prompt, verbose=True, output_key='description',memory=desc_memory)


# Define parent chain
parent_chain = SequentialChain(
    chains=[chain, chain2,chain3],
    input_variables=["name"],
    output_variables=["person", "dob","description"],
    verbose=True
)

# Execute parent chain if input text is provided
if input_text:
    st.write(parent_chain({'name': input_text}))
    
    with st.expander('person name'):
        st.info(person_memory.buffer)
    with st.expander('Major events'):
        st.info(desc_memory.buffer)    