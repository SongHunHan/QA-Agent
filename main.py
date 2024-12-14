import os
import json
import operator
from typing import Annotated, Sequence, TypedDict, Literal
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Initialize LLM and tools
llm = ChatOpenAI(model='gpt-4o-mini')
# tavily_tool = TavilySearchResults(max_results=2)

# Tool: 주제 요약
@tool
def summarize_to_topics(text: str) -> str:
    """
    Summarizes study content into high-level topics.
    """
    prompt = (
        "You are an AI assistant that helps summarize study content into concise, high-level study topics."
        "Each topic should represent a main idea of the text and serve as a useful entry point for deeper exploration.\n\n"
        f"Text:\n{text}\n\n"
        "You ''Must'' Return a JSON object with the following format:\n"
        "{'original_text': ..., 'topics': [...]}.\n"
    )
    result = llm.invoke(prompt)
    return result.content

# Tool: 질문 생성
@tool
def generate_questions(text: str) -> str:
    """
    Generates focused study questions based on summarized topics.
    """
    input_json = text.replace("'", '"')  # Ensure valid JSON format
    parsed_json = json.loads(input_json)

    prompt = (
        "You are an AI assistant that generates study questions based on provided topics"
        "Each question should be thought-provoking and aligned with the main themes of the topic.\n\n"
        "Questions should avoid being overly complex and should focus on a single clear idea. Generate one question per topic"
        f"Original Text:\n{parsed_json['original_text']}\n\n"
        f"Topics:\n{', '.join(parsed_json['topics'])}\n\n"
        "Generate one question for each topic in the format:\n"
        "1. Question ...\n2. Question ...\n\n"
        "You ''Must'' Return a JSON object with the format:\n"
        "{'questions': [...]}."
        "Additionally, the generated questions should be designed so that answers can be written in Korean."
    )
    result = llm.invoke(prompt)
    return result.content

# Query Node: Tool 기반 질문 생성 노드
def query_node(state):
    user_input = state['messages'][-1].content  # 사용자 입력
    # 1. Summarize topics
    summarized_topics = summarize_to_topics(user_input)
    # 2. Generate questions
    generated_questions = generate_questions(summarized_topics)
    return {'messages': [HumanMessage(content=generated_questions, name="Query")]}

# Judge Node: 답변 평가 노드
def judge_node(state):
    user_input = state['messages'][-1].content  # 사용자 답변
    prompt = (
        "You are an assistant that evaluates user answers. Provide:\n"
        "1. A score out of 10.\n"
        "2. Two strengths of the answer.\n"
        "3. Two areas for improvement.\n\n"
        f"User's Answer:\n{user_input}\n\n"
        "Return a JSON object with the format:\n"
        "{'score': ..., 'strengths': [...], 'improvements': [...]}."
        "Additionally, the generated strengths, improvements should be designed so that answers can be written in Korean."
    )
    result = llm.invoke(prompt)
    return {'messages': [HumanMessage(content=result.content, name="Judge")]}  # 수정된 접근 방식

# GeneralQuery Node: 일반 대화 처리
def general_query_node(state):
    user_input = state['messages'][-1].content  # 사용자 입력
    prompt = (
        "You are a friendly chatbot. Respond to the user input naturally and informatively.\n\n"
        f"User Input:\n{user_input}\n\n"
    )
    result = llm.invoke(prompt)  # LLM 호출
    return {'messages': [HumanMessage(content=result.content, name="GeneralQuery")]}

# Supervisor Node
members = ["Query", "Judge", 'GeneralQuery']
system_prompt = (
    "You are a supervisor managing agents: {members}. Based on the input:\n"
    "1. If the user input is a study-related topic, route to 'Query'.\n"
    "1-2. If questions have been successfully generated, return 'FINISH'.\n"
    "2. If the user input is a consists of a question and its corresponding answer related to the study material, route to 'Judge'.\n"
    "3. If the input is a general inquiry or conversational statement (e.g., greetings, introductions, or casual questions), route to 'GeneralQuery'.\n"
    "Return 'FINISH' when the workflow is complete."
)
options = ["FINISH"] + members

class RouteResponse(BaseModel):
    next: Literal["FINISH", "Query", "Judge", 'GeneralQuery']

supervisor_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Based on the conversation, decide the next agent: {options}.",
        ),
    ]
).partial(options=str(options), members=", ".join(members))

def supervisor_agent(state):
    supervisor_chain = supervisor_prompt | ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY).with_structured_output(RouteResponse)
    return supervisor_chain.invoke(state)

# Workflow Graph
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

workflow = StateGraph(AgentState)
workflow.add_node("Query", query_node)
workflow.add_node("Judge", judge_node)
workflow.add_node("GeneralQuery", general_query_node)
workflow.add_node("Supervisor", supervisor_agent)

# 연결 설정
for member in members:
    workflow.add_edge(member, "Supervisor")

conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END

workflow.add_conditional_edges("Supervisor", lambda x: x["next"], conditional_map)
workflow.add_edge(START, "Supervisor")

# 그래프 컴파일
memory = MemorySaver()
config = {"configurable": {"thread_id": 1}}
graph = workflow.compile(checkpointer=memory)

# Streamlit UI
st.set_page_config(page_title="스터디 Agent", layout="wide")
st.title("스터디 Agent")
st.markdown("Multi-Agent를 활용한 챗봇 (공부 내용 질문 생성, 사용자 답변 질문 평가)")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "오늘 공부한 내용을 입력해보세요!"}]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# User input
if prompt := st.chat_input():
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.spinner("Bot is typing..."):
        try:
            # Agent logic with graph.invoke
            state = {"messages": [HumanMessage(content=prompt)]}
            output = graph.invoke(state, config=config)
            bot_message = output["messages"][-1].content
            
            st.session_state["messages"].append({"role": "assistant", "content": bot_message})
            st.chat_message("assistant").write(bot_message)
        except Exception as e:
            st.error(f"An error occurred: {e}")
