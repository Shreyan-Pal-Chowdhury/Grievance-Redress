from setuptools import setup, find_packages

setup(
    name="GrievanceChatbot",
    version="1.0.0",
    author="Shreyan Pal Chowdhury",
    author_email="shreyan.palchowdhury@somnetics.in",
    description="AI-powered Consumer Grievance Chatbot with Groq Multimodal and MongoDB",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "flask",
        "pymongo",
        "groq",
        "langchain-community",
        "langchain-huggingface",
        "langchain-text-splitters",
        "faiss-cpu",
        "certifi"
    ],
    entry_points={
        "console_scripts": [
            "grievancebot=grievancechatbot.app:main"
        ],
    },
    python_requires=">=3.10",
)
