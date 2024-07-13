# 🤖 GRASP: Generative Retrieval Augmented Search Platform

![Screenshot (166)](https://github.com/user-attachments/assets/b1e15e48-97b7-4859-8434-5ebc72af45ca)

Welcome to GRASP! This platform allows you to explore various Retrieval-Augmented Generation (RAG) methods using different types of data sources, such as PDFs, web pages, text documents, audio files, databases, and APIs. Below you'll find a detailed guide on how to set up, use, and understand this project.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Home](#home)
  - [PDF RAG 📄](#pdf-rag-📄)
  - [Web RAG 🌐](#web-rag-🌐)
  - [Text Document RAG 📄](#text-document-rag-📄)
  - [Audio RAG 🎤](#audio-rag-🎤)
  - [Database RAG 🗄️](#database-rag-🗄️)
  - [API RAG 🔌](#api-rag-🔌)
- [Feedback](#feedback)
- [Contributing](#contributing)
- [License](#license)

## Installation

To set up this project, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/grasp.git
    cd grasp
    ```

2. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Set up environment variables:**
    - Create a `.env` file in the root directory of the project.
    - Add your OpenAI API key to the `.env` file:
      ```plaintext
      OPENAI_API_KEY=your_openai_api_key
      ```

4. **Run the Streamlit app:**
    ```bash
    streamlit run GRASP.py
    ```

## Usage

### Home
The Home page provides an overview and a warm welcome to the GRASP platform. You can navigate to different RAG methods from here.

### PDF RAG 📄
This page allows you to upload a PDF file and ask questions about its content.

1. **Upload a PDF file** using the sidebar.
2. **Enter your query** in the text input below.
3. **Click 'Get Results'** to process the PDF and retrieve relevant answers.

### Web RAG 🌐
This page allows you to enter a web URL and ask questions about its content.

1. **Enter a web URL** using the sidebar.
2. **Enter your query** in the text input below.
3. **Click 'Get Results'** to process the webpage and retrieve relevant answers.

### Text Document RAG 📄
This page allows you to upload a text document and ask questions about its content.

1. **Upload a text file** using the sidebar.
2. **Enter your query** in the text input below.
3. **Click 'Get Results'** to process the text document and retrieve relevant answers.

### Audio RAG 🎤
This page allows you to upload an audio file and ask questions about its transcribed content.

1. **Upload an audio file** using the sidebar.
2. **Enter your query** in the text input below.
3. **Click 'Get Results'** to process the audio file and retrieve relevant answers.

### Database RAG 🗄️
This page allows you to connect to a database, specify a table, and ask questions about its content.

1. **Enter database credentials** using the sidebar.
2. **Enter your query** in the text input below.
3. **Click 'Get Results'** to process the database and retrieve relevant answers.

### API RAG 🔌
This page allows you to specify an API endpoint and ask questions about its response.

1. **Enter the API URL** using the sidebar.
2. **Enter your query** in the text input below.
3. **Click 'Get Results'** to process the API response and retrieve relevant answers.

## Feedback
We value your feedback! Please provide your feedback using the text area in the sidebar and click 'Submit Feedback'.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
