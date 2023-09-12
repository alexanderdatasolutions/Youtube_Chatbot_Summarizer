# Youtube_Chatbot_Summarizer
Streamlit Interface for Bing Chilling Bot
This repository provides a simple yet powerful Streamlit-based interface for chatting with the Bing Chilling Bot. Leveraging the GPT-4 model, it's not just your average chatbot - it's built to provide richer and more contextually accurate responses based on previous conversations.

Features
YouTube Video Processing:

Easily process videos from YouTube links or directly from channels.
View the channel's recent videos and choose which ones to process.
Transcript Management:

View previous transcripts and summaries of YouTube videos.
Apply specific actions on these transcripts, such as summarization.
Dynamic Chat Experience with Bing Chilling Bot:

Refreshable bot knowledge.
Contextual chat, where the bot understands the topic of discussion and can leverage previous conversations for richer replies.
Prerequisites
Python 3.8 or higher
Streamlit
Pinecone
OpenAI (GPT-4 model)
Setup
Clone the repository:

bash
Copy code
git clone <repository-url>
cd <repository-dir>
Install the required packages:

bash
Copy code
pip install -r requirements.txt
Setup configuration:

Create a config.json in the root directory. This file should have the following structure:
json
Copy code
{
  "openai": {
      "key": "YOUR_OPENAI_API_KEY"
  },
  "pinecone": {
      "key": "YOUR_PINECONE_API_KEY",
      "env": "YOUR_PINECONE_ENV"
  }
}
Run the application:

bash
Copy code
streamlit run <filename>.py
How to use
Processing YouTube Videos:

Choose "Process New Videos".
Enter a YouTube link or a channel name.
If a channel name is provided, a list of recent videos will appear. Choose any video to process.
Click "Process" to process the entered link.
Viewing and Managing Transcripts:

Choose "Process Previous Transcripts".
Select a topic and then a channel to view the available transcripts.
Read raw transcripts and summaries, and perform actions on them.
Chat with Bing Chilling Bot:

Choose "Chat with Bing Chilling Bot".
Select a topic for the conversation.
Enter your query/question in the provided text area.
Click "Send" to see the bot's response.
Contribute
Feel free to fork, open issues, or submit PRs. All contributions are welcome!

License
MIT

Acknowledgements
Thanks to OpenAI for the powerful GPT-4 model and Pinecone for their state-of-the-art similarity search.
