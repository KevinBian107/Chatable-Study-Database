## Setting Up & Running "Chat" Function on Database
Create the conda enviornment by:
```python
conda env create
```

If you want to use GPT with API, you need to create your own OpenAI account and then embed your API key in your system with writing this in your `.bash` file:

```python
export OPENAI_API_KEY = "your api key"
```

Run the following to update system file:

```python
source ~/.bash_profile
```

Enter the conda environment

```python
conda activate ucsd_study
```

Then run an instance (`chat_with_feedback`) of our chat function by:

```python
python chat/chat_with_feedback.py
```

We have created a few versions of our chat functions:
- `chat_base.py` is the vanill implementation of the chat function.
- `chat_langchain.py` atampts to us  the langchain package (*not working yet*).
- `chat_standard.py` is the currently useful standard version.
- `chat_with_feedcack.py` is `chat_standard.py` but implemented a feedcak for follow up questions, which is much smarter and useful than the standard version.

An example of chat feedback in in [here](https://github.com/KevinBian107/Kaiwen-Study-Database/tree/main/logs) and we have a demo of chat function in here:

<div style="width: 100%; display: flex; flex-direction: column; align-items: center;">
  <video controls autoplay style="width: 80%; height: auto;" muted>
    <source src="../demos/chat/live_chat.mp4" type="video/mp4">
      Your browser does not support the video tag.
  </video>
</div>