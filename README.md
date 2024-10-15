# Chat with documentation

This Python project that allows you to chat with a chatbot about the PDF you uploaded. and generate a PDF transcript of the conversation. The project is built using Python and Streamlit framework.

This is a fork from https://github.com/sudan94/chat-pdf-hugginface, however it is heavily changed to fit REST api communication with the Gemini model from google. 
```
A `conda` environment is used instead of `venv` virtual environment

## Installation

To run this project, please follow the steps below:

1. Clone the repository:

```shell
git clone git@github.com:
cd chat-pdf-hugginface
```

2. Create and activate a conda virtual environment (optional but recommended):

```shell
conda create -n SNADSenv python=3.10
conda activate SNADSenv
```

3. Install the dependencies from the `requirements.txt` file:

```shell
pip install -r requirements.txt
```

4. You will need a GENERATIVE_API_KEY for this next step. To obtain one for free, go to https://aistudio.google.com/app/apikey. Create a New token. Then, change the file in this directory, name is `.env` and enter `GENERATIVE_API_KEY = "token"`,  replacing `token` with your User Access Token. Save the `.env` file. The `.gitignore` file will ignore the `.env` for git operation.

## Running the Project

Once you have installed the required dependencies, you can run the project using Streamlit, which should have been installed with `requirements.txt`. Streamlit provides an easy way to create interactive web applications in Python.

To start the application, run the following command:

```shell
streamlit run app.py
```

This will start the Streamlit server and open the application in your default web browser..  

### Git flow
`.gitit.sh` can be used for easy pushing updates to remote repo.  

Activate with:  
```shell
chmod +x .gitit.sh
```
Then, to add, commit, and push to remote repo:
```
./gitit.sh
```

## Possible error

The document that you upload is chached for 20 minutes. If you want to use the chat more than 20 minutes after uploading the content, either change this value in the code or restart the application (restarting will mean you lose the conversation history).

## License

This project is licensed under the [MIT License](LICENSE).

