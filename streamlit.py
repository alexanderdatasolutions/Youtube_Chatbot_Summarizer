# First, import Streamlit
import streamlit as st
import os
import openai
from pydub import AudioSegment
from tqdm import tqdm
from pytube import YouTube
from pytube import Channel
import json
import time
import streamlit as st
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
import os
import json
from langchain.memory import ChatMessageHistory
import pinecone
from langchain.chat_models import ChatOpenAI
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

with open('config.json') as f:
    config = json.load(f)

os.environ['OPENAI_API_KEY'] = config["openai"]["key"]
openai.api_key = config["openai"]["key"]
os.environ['PINECONE_API_KEY'] = config["pinecone"]["key"]
os.environ['PINECONE_API_ENVIRONMENT'] = config["pinecone"]["env"]

def gather_intent(user_message, conversation_history):
    chat = ChatOpenAI(openai_api_key = config["openai"]["key"], model_name = 'gpt-4')
    combined_input = user_message + str(conversation_history)

    # Query the llm model to gather the intent of the user based on the combined input
    intent = chat.predict("What is the user really asking about here"+combined_input)
    return intent

def reload_and_reparse_database():
    # Load the openai key
    with open('config.json') as f:
        config = json.load(f)

    os.environ['OPENAI_API_KEY'] = config["openai"]["key"]

    # Initialize pinecone
    pinecone.init(
        api_key=config['pinecone']['key'],  
        environment=config['pinecone']['env']  
    )

    index_name = "bingchillingbot"
    index = pinecone.Index(index_name)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap  = 100,
        length_function = len,
        is_separator_regex = False,
    )

    # Path to root directory where collection folders are
    root_path = "."
    ignore_list = [ "_pycache__"]

    # Iterate through each collection in the root directory while ignoring the ignore list
    for collection in os.listdir(root_path):
        collection_path = os.path.join(root_path, collection)
        
        # Check if it's a directory and not in ignore list
        if os.path.isdir(collection_path) and collection not in ignore_list:

            # Clear the collection from Pinecone before reloading
            index.delete(delete_all=True, namespace=collection)

            # Iterate through each person's folder in the collection
            for person in os.listdir(collection_path):
                person_path = os.path.join(collection_path, person)
                
                if os.path.isdir(person_path):
                    # Iterate through each base directory (Raw_Transcripts and Summaries)
                    for base_folder in ['Raw_Transcripts']:
                        base_folder_path = os.path.join(person_path, base_folder)
                        
                        if os.path.exists(base_folder_path):
                            # Process each text file in the base directory
                            for file in os.listdir(base_folder_path):
                                if file.endswith(".txt"):
                                    file_path = os.path.join(base_folder_path, file)
                                    
                                    # Load the document, split it into chunks, embed each chunk and load it into the vector store
                                    raw_documents = TextLoader(file_path).load()
                                    documents = text_splitter.split_documents(raw_documents)
                                    db = Pinecone.from_documents(documents, OpenAIEmbeddings(), index_name=index_name,namespace=collection)

def export_chunk_to_temp(chunk):
    chunk.export("temp_chunk.wav", format="wav",
                 codec="pcm_s16le", parameters=["-ar", "44100"])

def get_chunk_size(chunk):
    export_chunk_to_temp(chunk)
    size = os.path.getsize("temp_chunk.wav")
    os.remove("temp_chunk.wav")
    return size

def split_text(text, max_length=2000):
    words = text.split()
    text_chunks = []
    current_chunk = []

    for word in words:
        if len(" ".join(current_chunk + [word])) <= max_length:
            current_chunk.append(word)
        else:
            text_chunks.append(" ".join(current_chunk))
            current_chunk = [word]

    if current_chunk:
        text_chunks.append(" ".join(current_chunk))

    return text_chunks

def split_audio(audio_file_path, max_size_bytes=25 * 1024 * 1024, fixed_duration_seconds=60):
    audio = AudioSegment.from_file(audio_file_path)
    total_duration = len(audio) / 1000
    chunks = []
    start_time = 0
    end_time = 0

    while end_time < total_duration:
        end_time = min(end_time + fixed_duration_seconds, total_duration)
        chunk = audio[start_time * 1000:end_time * 1000]

        while get_chunk_size(chunk) > max_size_bytes:
            fixed_duration_seconds -= 1
            end_time = start_time + fixed_duration_seconds
            chunk = audio[start_time * 1000:end_time * 1000]

        chunks.append(chunk)
        start_time = end_time

    return chunks

def split_and_summarize_text(text):
    # Split text
    text_chunks = split_text(text)

    # Initialize the Streamlit progress bar
    progress_bar = st.progress(0)
    num_chunks = len(text_chunks)
    
    summarized_chunks = []

    # Summarize each chunk
    for idx, chunk in enumerate(text_chunks):
        summarized_chunks.append(summarize_chunk(chunk))
        
        # Update the Streamlit progress bar
        progress_bar.progress((idx + 1) / num_chunks)

    return " ".join(summarized_chunks)

def summarize_chunk(chunk, model="gpt-4"):
    system_content = 'Extract the most useful information from the following text, this is for further summarization, so be detailed:'
    summary = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": system_content + chunk}
        ]
    )
    return summary["choices"][0]["message"]["content"]

def transcribe_video(audio_file_path, video_channel, video_title, topic):
    # Split audio into chunks and transcribe
    audio_chunks = split_audio(audio_file_path)
    total_chunks = len(audio_chunks)
    st.write(f"Total chunks: {total_chunks}")
    transcripts = []
    
    progress_bar_transcribe = st.progress(0)  # New progress bar for transcription
    for index, chunk in enumerate(audio_chunks):
        export_chunk_to_temp(chunk)
        with open("temp_chunk.wav", "rb") as audio_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
            transcripts.append(transcript["text"])
        os.remove("temp_chunk.wav")
        progress_bar_transcribe.progress((index + 1) / total_chunks)
    full_transcription = " ".join(transcripts)
    
    # Save transcription under the topci / video channel folder / Raw_Transcripts folder
    if not os.path.exists(topic + '/' + video_channel + "/Raw_Transcripts"):
        os.makedirs(topic + '/' + video_channel + "/Raw_Transcripts")
    with open(topic + '/' + video_channel + "/Raw_Transcripts/" + video_title + "_raw.txt", "w") as file:
        file.write(full_transcription)
    
    return full_transcription

def extract_audio_from_video(video_file):
    video = AudioSegment.from_file(video_file, format="mp4")
    video.export("temp_audio.wav", format="wav")

def download_video(link, topic):
    try:
        youtubeObject = YouTube(link).streams.get_audio_only()
        # Set the file name to the video title
        video_channel = YouTube(link).author
        video_title = youtubeObject.title
        # Create a folder for the video channel if it doesn't exist
        if not os.path.exists(topic + '/' + video_channel):
            os.makedirs(topic + '/' + video_channel)
        # Check if the audio has already been downloaded
        if os.path.exists(topic + '/' +video_channel + "/" + video_title):
            print("Video already downloaded")
            return video_title, video_channel
        # Download the video
        youtubeObject.download(output_path=topic + '/' +video_channel , filename="temp")

        print('video title: ', video_title)
    except Exception as e:
        print(f"An error has occurred: {e}")
    return video_title, video_channel

def process_youtube(link, topic):
    # Overall Progress Bar
    st.write("Processing YouTube video...")
    progress_bar = st.progress(0)

    # Download the video
    st.write("Downloading video...")
    video_title, video_channel = download_video(link, topic)
    video_title = sanitize_filename(video_title)
    st.write("Video downloaded successfully!")
    progress_bar.progress(20)
    # the path is now the channels name as the folder
    extract_audio_from_video(topic + '/' + video_channel +'/'+ "temp")

    st.write("Transcribing video...")
    transcribe_video("temp_audio.wav", video_channel, video_title, topic)
    progress_bar.progress(40)
    st.write("Video transcribed successfully!")

    st.write("Summarizing video...")
    with open(topic + '/' + video_channel + "/Raw_Transcripts/" + video_title + "_raw.txt", "r") as file:
        raw_transcript = file.read()
    summarized_transcript = split_and_summarize_text(raw_transcript)
    # Save summary under the video channel folder / Summaries folder
    if not os.path.exists(topic + '/' +video_channel + "/Summaries"):
        os.makedirs(topic + '/' +video_channel + "/Summaries")
    
    with open(topic + '/' + video_channel + "/Summaries/" + video_title + "_summary.txt", "w") as file:
        # Write the summary to the .txt file named after the video title_summary.txt
        file.write(summarized_transcript)
    progress_bar.progress(80)
    st.write("Video summarized successfully!")
    
    os.remove(topic + '/' +video_channel + "/" + "temp")
    os.remove("temp_audio.wav")
    progress_bar.progress(100)
    st.write("Video processed successfully!")
    
def sanitize_filename(filename):
    invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

def summarize_transcript(transcript):
    return split_and_summarize_text(transcript)

def process_multiple_youtube_links(links, topic):
    for link in links:
        process_youtube(link, topic)

def split_into_chunks(text, max_length):
    """
    Split a text into chunks of max_length.
    """
    chunks = []
    while len(text) > max_length:
        # Find the last whitespace within the max_length
        split_idx = text.rfind(' ', 0, max_length)
        
        # If we couldn't find a whitespace, split at max_length
        split_idx = split_idx if split_idx > 0 else max_length
        
        chunks.append(text[:split_idx])
        text = text[split_idx:]
    
    chunks.append(text)
    return chunks

def perform_action_on_transcript(transcript, action, model):
    MAX_LENGTH = 10000
    
    if len(transcript) <= MAX_LENGTH:
        # Call OpenAI directly with user's action and return
        return action_on_whole_transcript(transcript, action, model)

    # If transcript is too long
    while len(transcript) > MAX_LENGTH:
        chunks = split_into_chunks(transcript, MAX_LENGTH)
        # Create a progress bar for the chunks
        progress_bar = st.progress(0)
        num_chunks = len(chunks)
        
        summarized_chunks = []
        for idx, chunk in enumerate(chunks):
            # Log the chunk before summarization
            log_content(f"Original Chunk {idx + 1}", chunk)
            
            # Summarize the chunk
            summarized_chunk = summarize_chunk(chunk, model)
            summarized_chunks.append(summarized_chunk)
            
            # Log the chunk after summarization
            log_content(f"Summarized Chunk {idx + 1}", summarized_chunk)
            
            # Update the progress bar
            progress_bar.progress((idx + 1) / num_chunks)

        # Combine the summaries and re-check the length
        transcript = ' '.join(summarized_chunks)

    # Log the final transcript before applying the user's action
    log_content("Final Transcript", transcript)
    # Once the transcript is below the max length, apply the user's action
    final_result = action_on_whole_transcript(transcript, action, model)
    
    return final_result

def action_on_whole_transcript(transcript, action, model):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": action},
            {"role": "user", "content": action + transcript}
        ]
    )
    return response["choices"][0]["message"]["content"]

def log_content(content_name, content):
    """Utility function to log content."""
    log_message = f"{content_name}:\n{content}\n{'-'*50}"
    # output to a file
    with open("log.txt", "a") as file:
        file.write(log_message)

def get_video_links(channel_name, number):
    options = webdriver.ChromeOptions()
    options.add_experimental_option("detach", True)
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-notifications")
    options.add_argument("--disable-Advertisement")
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--headless")
    
    s = Service('./chromedriver.exe')
    driver = webdriver.Chrome(service=s, options=options)
    driver.get(f'https://www.youtube.com/{channel_name}/videos')
    time.sleep(3)
    
    SCROLL_PAUSE_TIME = 1
    last_height = driver.execute_script("return document.documentElement.scrollHeight")
    
    data = []
    
    while len(data) < number:
        driver.execute_script("window.scrollTo(0,document.documentElement.scrollHeight);")
        time.sleep(SCROLL_PAUSE_TIME)
        new_height = driver.execute_script("return document.documentElement.scrollHeight")
        
        if new_height == last_height:
            break
        last_height = new_height
        
        elements = WebDriverWait(driver, 20).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'div#details')))
        for e in elements:
            title = e.find_element(By.CSS_SELECTOR, 'a#video-title-link').get_attribute('title')
            vurl = e.find_element(By.CSS_SELECTOR, 'a#video-title-link').get_attribute('href')
            views = e.find_element(By.XPATH, './/*[@id="metadata"]//span[@class="inline-metadata-item style-scope ytd-video-meta-block"][1]').text
            date_time = e.find_element(By.XPATH, './/*[@id="metadata"]//span[@class="inline-metadata-item style-scope ytd-video-meta-block"][2]').text
            data.append({
                'video_url': vurl,
                'title': title,
                'date_time': date_time,
                'views': views
            })
            if len(data) >= number:
                break
                
    driver.quit()
    return [item['video_url'] for item in data][:number]


# Streamlit Interface
def main():
    # Set wide mode to default
    st.set_page_config(layout="wide")
    st.title("Bing Chilling Bot")

    # Load the openai key
    with open('config.json') as f:
        config = json.load(f)

    os.environ['OPENAI_API_KEY'] = config["openai"]["key"]

    # Initialize pinecone
    pinecone.init(
        api_key=config['pinecone']['key'],  
        environment=config['pinecone']['env']  
    )
    chatbot = ChatOpenAI(openai_api_key = config["openai"]["key"], model_name = 'gpt-4')
    index_name = "bingchillingbot"
    embeddings = OpenAIEmbeddings(openai_api_key=config["openai"]["key"])
    db = Pinecone.from_existing_index(index_name, embeddings)
    conversation_history = ChatMessageHistory()

    # Create a slider in the top middle of the page for Process New Videos or Process Previous Transcripts
    # create a sidebar for this
    st.sidebar.title("What do you want to do?")
    process_type = st.sidebar.radio("", ("Process New Videos", "Process Previous Transcripts", "Chat with Bing Chilling Bot"))


    if process_type == "Process New Videos":
        # Text box for user to enter YouTube link
        link = st.text_input("Enter the YouTube link:")
        topic = st.text_input("Enter the topic:")
        channel_name = st.text_input("Enter the channel name:")
        
        if channel_name:
            video_links = get_video_links(channel_name, 10)
            # Buttone to process all videos in the list
            if st.button("Process All Videos"):
                process_multiple_youtube_links(video_links, topic)
                st.success("All videos processed successfully!")

            for video_link in video_links:
                video_details = YouTube(video_link)
                st.image(video_details.thumbnail_url, width=200, caption=video_details.title)
                st.write("Process video:")
                if st.button("Process", key=video_link):
                    process_youtube(video_link, topic)
                    st.success("Video processed successfully!")
                    processed_video = sanitize_filename(YouTube(video_link).title)
                    st.write(f"Processed video: {processed_video}")
                st.markdown("---")

        # Button to process the link
        if st.button("Process"):
            if link:
                process_youtube(link, topic)
                st.success("Video processed successfully!")
                processed_video = sanitize_filename(YouTube(link).title)
                st.write(f"Processed video: {processed_video}")
            else:
                st.warning("Please enter a valid YouTube link.")
        
    if process_type == "Process Previous Transcripts":
        # Sidebar for selecting channels and videos
        
        ignored_folders = ["__pycache__", ".git", ".vscode", "summarizer"]
        topics = [folder for folder in os.listdir() if os.path.isdir(folder) and folder not in ignored_folders]
        selected_topic = st.selectbox("Select a topic:", topics)

        # the channel list are in the folder of the topics the user selects
        channel_list = [folder for folder in os.listdir(selected_topic) if os.path.isdir(os.path.join(selected_topic, folder))]

        selected_channel = st.selectbox("Select a channel:", channel_list)
        selected_video = ""
        if selected_channel:
            # The transcripts for the videos are stored in the Raw_Transcripts folder
            video_list = [file.rsplit("_raw.txt", 1)[0] for file in os.listdir(os.path.join(selected_topic, selected_channel, "Raw_Transcripts")) if file.endswith("_raw.txt")]

            selected_video = st.selectbox("Select a video:", video_list)

        if selected_video:
            # Load and display raw transcript and summary
            raw_transcript_path = os.path.join(selected_topic, selected_channel, 'Raw_Transcripts', f"{selected_video}_raw.txt")
            with open(raw_transcript_path, "r") as file:
                raw_transcript = file.read()

            summary_transcript_path = os.path.join(selected_topic, selected_channel, 'Summaries', f"{selected_video}_summary.txt")
            try:
                with open(summary_transcript_path, "r") as file:
                    summary_transcript = file.read()
            except FileNotFoundError:
                # Some videos may not have a summary yet
                summary_transcript = "Summary not available yet."

            # create an expander
            with st.expander("Raw Transcript"):
                st.write(raw_transcript)
            with st.expander("Summary"):
                st.write(summary_transcript)

            # Ask user for action to perform on raw transcript or summary
            action = st.text_area("Enter an action to perform on the text:")
            # Make the user select the raw transcript or summary
            text_to_process = st.radio("Select text to process:", ("Raw Transcript", "Summary"))
            if st.button("Perform action"):
                if action:
                    selected_text = raw_transcript

                    with st.spinner("Performing action on text..."):
                        final_result = perform_action_on_transcript(selected_text, action, "gpt-4")
                    st.success("Action performed successfully!")

                    # Log the content after the action is performed
                    log_content("Processed " + text_to_process, final_result)
                    st.write(final_result)
                    
                else:
                    st.warning("Please enter a valid action.")

    if process_type == "Chat with Bing Chilling Bot":
        st.header("Chat with Bing Chilling Bot!")
        
        # Create a drop-down for the user to select the topic which is going to be a set of root folders
        ignored_folders = ["__pycache__", ".git", ".vscode", "summarizer"]
        topics = [folder for folder in os.listdir() if os.path.isdir(folder) and folder not in ignored_folders]
        selected_topic = st.selectbox("Select a topic:", topics)
        st.write(f"Selected topic: {selected_topic}")
        # refresh bot knowledge button which runs the reload_and_reparse_database function
        if st.button("Refresh Bot Knowledge"):
            reload_and_reparse_database()
            st.success("Bot knowledge refreshed successfully!")
        # store it in a session state to be passed to the similarity search later
        st.session_state['selected_topic'] = selected_topic
        # Line spacer
        st.markdown("---")

        # Create a text area for the user to enter their action (query)
        query = st.text_area("Ask it a question:")

        # If the user presses the "Send" button
        if st.button("Send"):
            if query:
                # Append the user's query to session state's conversation history
                st.session_state.conversation_history.append({"role": "User", "message": query, "timestamp": time.time()})
                
                # Add the user's question to the conversation history
                conversation_history.add_user_message(query)

                # Gather any previous conversation history for the user and ai
                chat_history = conversation_history.messages

                # Gather the intent from the user's question
                intent = gather_intent(query, chat_history)

                # Combine the user question, conversation history, and intent into a single prompt
                prompt = query + str(chat_history) + intent

                # Get the context from the chat history - We only want scores in the metadata of .8 or higher
                context = db.similarity_search(query, namespace=st.session_state['selected_topic'], k=13) # We're using the selected topic here as the namespace

                # Extracting page contents
                page_contents = [doc.page_content for doc in context]

                prompt = f'''Here is the users question: {query} \n +
                        Here is the previous conversation history: {chat_history} \n +
                        Here is what the user is really asking about: {intent} \n +
                        Use this information in order to answer it: {page_contents} \\n '''


                # Log the prompt
                log_content("Prompt", prompt)

                # Generate the ai's response
                ai_response = chatbot.predict(prompt)

                # Append the bot's reply to session state's conversation history
                st.session_state.conversation_history.append({"role": "Bot", "message": ai_response, "timestamp": time.time()})

                # print the bot's response
                st.write(f"Bing Chilling Bot: {ai_response}")




if __name__ == "__main__":
    main()