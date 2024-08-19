import streamlit as st
import random
import requests
from dotenv import load_dotenv
import os
import io
from PIL import Image
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()

# Set the Hugging Face API tokens
API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
if not API_TOKEN:
    st.error("HUGGINGFACE_API_TOKEN is not set in the environment.")
    st.stop()

# Set the Hugging Face Inference API URL for the image generation model
API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

# Initialize the InferenceClient with the Phi-3 model
client = InferenceClient(
    model="microsoft/Phi-3-mini-4k-instruct",
    token=API_TOKEN
)

# Define the list of game links
game_links = [
    "https://richardhenyash.github.io/balloon-pop-maths/",
    "https://simplecoder12.github.io/game-1/",
    "https://jisll.github.io/MathGame/"
]

def show_random_game_link():
    # Choose a random link from the list
    selected_link = random.choice(game_links)
    
    # Embed the selected link in an iframe
    st.components.v1.html(
        f"""
        <iframe src="{selected_link}" width="100%" height="600" style="border:none;"></iframe>
        """,
        height=600,
    )

# FAQ contents
faq_contents = {
    "**What is SWS chat**": "<h3>SWS Chat</h3><p>It is an advanced conversational AI developed by SWS Technologies, designed to simulate human-like interactions through natural language processing. Modeled after ChatGPT, a widely recognized AI language model, SWS Chat aims to provide intelligent and engaging conversations across a variety of applications.</p>",
    "**How to use SWS chat**": "<h3>Introduction</h3><p>SWS is your AI-powered chatbot designed to assist with a wide range of queries and tasks. Whether you need information, support, or just a friendly chat, SWS is here to help.</p><h3>Starting a Conversation</h3><p>Click or tap on the chat widget to initiate a conversation.</p><p>You can start by typing a greeting or directly asking a question or stating a task you need help with.</p>",
    "**What is SWS chat made on**": "<p>SWS chat is formed on the basis of new generative AI like <code>PHI-3</code> and <code>FLUX_SCHNELL</code></p>",
    "**Key features of SWS chat**": "<h3>Natural Language Understanding</h3><p>SWS Chat is equipped with sophisticated algorithms to comprehend and generate human-like responses. It can engage in diverse topics, from casual chit-chat to complex discussions, making it versatile for numerous use cases.</p><h3>Adaptability</h3><p>Whether used for customer support, educational purposes, or personal assistance, SWS ChatGPT can be tailored to meet specific needs. Its adaptability allows it to handle a range of tasks, from answering queries to assisting with problem-solving.</p><h3>Learning and Improvement</h3><p>Like its predecessors, SWS ChatGPT benefits from continuous learning and updates, improving its responses and expanding its knowledge base over time.</p>",
    "**Features and Capabilities**": "<h3>Answering Questions</h3><p>Ask SWS any question you have, whether it’s about general knowledge, specific information, or how-to guides.</p><p>Example: “Which is the longest river in India?” or “What is photosynthesis?”</p><h3>Providing Support</h3><p>If you need assistance with technical issues or customer service, SWS can guide you through troubleshooting steps or direct you to the appropriate resources.</p><p>Example: “I am having trouble finding out the answer to this question.”</p><h3>Casual Conversation</h3><p>Feel free to chat with SWS about various topics. It’s programmed to handle casual conversations and provide engaging interactions.</p><p>Example: “Tell me a joke.”</p>",
    "**Tips for Effective Use**": "<h3>Be Specific</h3><p>The more specific you are with your queries, the better SWS can assist you.</p><p>For example, instead of asking “Tell me about history,” specify “Tell me about the Industrial Revolution.”</p><h3>Provide Feedback</h3><p>If you encounter issues or have suggestions for improvement, use the feedback option (if available) to help enhance SWS’s capabilities.</p><h3>Privacy and Security</h3><p>Avoid sharing sensitive personal information in your conversations. SWS is designed to prioritize your privacy, but it's always best to be cautious.</p>",
    "**How to create an image with SWS CHAT**": "<h3>SWS CHAT is an innovative bot designed to make image creation simple and efficient.</h3><p>Whether you're a designer, content creator, or just someone looking for a quick way to create visuals, SWS CHAT has you covered. Follow these steps to generate your desired images:</p><h3>Step 1: Start a Conversation with SWS CHAT</h3><p>Begin by initiating a chat with SWS CHAT. You can do this through the platform where the bot is integrated (e.g., a website, messaging app, or other interface).</p><h3>Step 2: Provide a Detailed Description</h3><p>Tell SWS CHAT what kind of image you need. Be as detailed as possible to ensure the bot understands your vision. For example, you can specify:</p><ul><li>Subject: Describe what you want in the image (e.g., 'A sunset over a mountain range').</li><li>Style: Indicate the artistic style you're looking for (e.g., 'realistic,' 'cartoonish,' 'minimalistic').</li><li>Colors: Mention any specific colors you want to be included or avoided.</li><li>Mood/Theme: Describe the overall feel of the image (e.g., 'calm and serene,' 'vibrant and energetic').</li></ul><h3>Step 3: Customize Further (Optional)</h3><p>If SWS CHAT offers customization options, you can tweak additional settings such as image dimensions, aspect ratio, or any special effects you might want to apply.</p><h3>Step 4: Generate the Image</h3><p>Once you've provided all the necessary details, simply instruct SWS CHAT to generate the image. The bot will process your input using its AI algorithms and create a visual that matches your description.</p><h3>Step 5: Review and Download</h3><p>After a short wait, SWS CHAT will present you with the generated image. You can review it to ensure it meets your expectations. If satisfied, you can download the image directly for immediate use. If not, you can tweak your description and generate a new image.</p><h3>Step 6: Share or Save</h3><p>Once you've downloaded your image, it's ready to be used however you see fit. Whether you want to share it on social media, include it in a presentation, or print it out, SWS CHAT makes it easy to bring your creative ideas to life.</p>",
    "**Troubleshooting**": "<h3>If SWS Doesn’t Understand Your Query</h3><p>Try rephrasing your question or providing more context.</p><h3>Technical Issues</h3><p>If you experience problems accessing SWS or its features, check your internet connection or refresh the page. If issues persist, contact support.</p>",
}

# Streamlit app configuration
st.set_page_config(page_title="SWS CHATBOT", page_icon="💬")

# Main title and sidebar setup
st.markdown(
   """ 
       <h1>
        <span style='font-weight: bold; font-size: 50px;'>💬SWSSSSSS</span>
        </h1>
   """,
   unsafe_allow_html=True 
)

with st.sidebar:
    st.markdown(
       """
    <h1>
        <span style='font-weight: bold; font-size: 40px;'>SWS CHAT</span> 
        <span style='font-size: 30px;'><span style='font-size: 40px;'>(S</span>mart 
        <span style='font-size: 40px;'>W</span>eb 
        <span style='font-size: 40px;'>S</span>ervices</span>
        <span style='font-size: 40px;'>)</span>
    </h1>
    """,
   unsafe_allow_html=True
    )
    selected_faq = st.sidebar.radio("Select a FAQ", list(faq_contents.keys()))
    st.sidebar.markdown(
        """
        <span style='font-weight: bold; font-size: 20px;'>Answer:</span>
        """,
        unsafe_allow_html=True
    )
    st.sidebar.markdown(faq_contents[selected_faq], unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
prompt = st.chat_input("What is up?")

if prompt:
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    def check_image_command(prompt):
        if prompt.startswith('/image'):
            return prompt[7:]  # Remove the "/image " part
        return False
    
    def check_game_command(prompt):
        if prompt.strip().lower() == '/game':
            return True
        return False

    # Check if the prompt is a game command
    if check_game_command(prompt):
        show_random_game_link()
        st.session_state.messages.append({"role": "assistant", "content": "Loading a random game for you..."})
    else:
        image_prompt = check_image_command(prompt)

        if image_prompt:
            try:
                # Generate image output using Hugging Face API
                data = {"inputs": image_prompt}
                response = requests.post(API_URL, headers=headers, json=data)
                response.raise_for_status()  # Raise an error for bad responses
                image_bytes = response.content

                # Load and display the image
                image = Image.open(io.BytesIO(image_bytes))
                with st.chat_message("assistant"):
                    st.image(image, caption="Generated Image")
                st.session_state.messages.append({"role": "assistant", "content": "![Image](generated_image.png)"})

                # Optionally, you can save the image to a file
                image.save("generated_image.png")
            except Exception as e:
                st.error(f"Error generating image: {e}")
        else:
            response = ""
            try:
                # Generate text output using Hugging Face Phi-3 model
                chat_messages = [{"role": "user", "content": msg["content"]} for msg in st.session_state.messages]
                chat_messages.append({"role": "user", "content": prompt})

                for message in client.chat_completion(
                    messages=chat_messages,
                    max_tokens=500,
                    stream=True,
                ):
                    response += message.choices[0].delta.content
                
                # Display assistant response in chat message container
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error generating text: {e}")
