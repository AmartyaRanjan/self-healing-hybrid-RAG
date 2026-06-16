import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class HyDEGenerator:
    def __init__(self):
        print("[*] Initializing Phase 3: HyDE (Hypothetical Document Embedding) Generator...")
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.7, # Higher temperature for creative "ideal" answers
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert technical writer. Write a detailed, hypothetical paragraph that answers the user's question. This paragraph will be used to improve vector search retrieval."),
            ("human", "{question}")
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()

    def generate_hypothetical_document(self, question: str) -> str:
        print(f"[*] HyDE: Generating idealized answer for query optimization...")
        return self.chain.invoke({"question": question})
