### Quora-Insincere-Questions-Classification  
---  

This is source code for the Kaggle competition, [Quora Insincere Questions Classification](https://www.kaggle.com/c/quora-insincere-questions-classification). 

**Goal:** Detect toxic content to improve online conversations in Quora platform.

### Installation:

To run this notebook interactively:

1. Download this repository in a zip file by clicking on this [link](https://github.com/PacoPacov/insincere_questions_classification.git) or execute this from the terminal:
`git clone https://github.com/PacoPacov/insincere_questions_classification.git`
2. Install virtual environment via one of:
    * virtualenv
    * venv
    * anaconda env
3. Navigate to the directory where you unzipped or cloned the repo and create a virtual environment.
4. Activate the environment with `source env/bin/activate`
5. Install the required dependencies with `pip install -r requirements.txt`.
6. Add the path to the repo into your PYTHONPATH.
7. To use the data you need to run set_train_data.sh
```sh
>>> sh set_train_data.sh
```
8. Execute `jupyter notebook` or `jupyter lab` from the command line or terminal.
9. Click on `notebooks` folder on the Jupyter Notebook dashboard, select Data Exploration and enjoy!
10. When you're done deactivate the virtual environment with `deactivate`.

### Usage:
You can use already trained model in two ways:  
* Inside python terminal (like the default one or ipython):  
```python  
>>> from insincere_questions_classification import make_prediction  
>>> make_prediction("This is a simple question?")
    1
```  
* Through command line (You open cmd/terminal):  
```sh
>>> python insincere_questions_classification/main/predict_sents.py --text "This is a simple question?"
The text was evaluated with the value:  1
```
* Last but not least through Flask Web API (![logo](http://www.clker.com/cliparts/8/2/4/d/1197124191544202843dchandlr_dchandlr_work.svg.svg.thumb.png))  
To start the application you need to run this command:  
```sh
>>> python insincere_question_classification/web_app/flaskblob.py
```