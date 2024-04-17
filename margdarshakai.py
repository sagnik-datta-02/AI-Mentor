from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from dotenv import load_dotenv
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import os
from langchain.agents import create_structured_chat_agent
import langchain_google_vertexai as verai
from langchain.agents import AgentExecutor
from langchain.agents import create_tool_calling_agent
from langchain.prompts import PromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate
)
from langchain.globals import set_verbose

set_verbose(True)

def callmensa(input_text):
    api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=600)
    wiki=WikipediaQueryRun(api_wrapper=api_wrapper)
    
    search = TavilySearchAPIWrapper(tavily_api_key=os.getenv("TAVILY_API_KEY"))
    tavily= TavilySearchResults(api_wrapper=search)
    
    
    tools=[tavily,wiki]
    
    from langchain_fireworks import ChatFireworks
    apikey=os.getenv('FIREWORKS_API_KEY')
   
    llm = ChatFireworks(model="accounts/fireworks/models/mixtral-8x7b-instruct", api_key=apikey)
   
   # llm = ChatGoogleGenerativeAI(model="gemini-pro",
   #                          temperature=0.1, safety_settings={
    #    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    #   HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
   #}, google_api_key=GOOGLE_API_KEY)

    from langchain.tools.render import render_text_description
    from langchain_core.output_parsers import JsonOutputParser
    rendered_tools = render_text_description(tools)
    system_prompt = f"""You are an Mentor assistant for personalized career guidance and professional development based on personality. 
You are specialized on questions about various personality types, career advice,latest industry insights, and skill-building exercises
 You will not entertain any questions not related to personality types, career advice,latest industry insights, and skill-building exercises , you will answer 'I am a Mentor AI assistant, I do not have the asnwer for it currently', if anything else is asked.
  You will not search wikipedia or tavily on any other topics expect : [personality types, career advice,latest industry insights, skill-building exercises, networking oppurtunities].You have access to the following set of tools.
  You will not give the option to user to search a query in any tools mentioned apart from the topics you cater to.
  Here are the names and descriptions for each tool:

{rendered_tools}


"""

    prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("user", "{input}")]
)


    chain = prompt | llm 
    answer=chain.invoke({"input": {input_text}})
    print(answer)
    return(answer.content)



def main():
    st.set_page_config("MargDarshak")
    st.header("Chat with MargDarshak powered by Langchain 💁")

    user_question = st.text_input("Ask any Queries about Personality , Career Suggestions etc")

    if user_question:
        st.write(callmensa(user_question))

if __name__ == "__main__":
    main()
