


## Document Analysis, Classification, and Summarization Project

### Overview
This project demonstrates various **Natural Language Processing (NLP)** techniques using the popular **20 Newsgroups dataset**. The dataset contains documents across different topics such as sports, automobiles, medicine, and electronics. The project covers several NLP tasks, including **entity recognition**, **document classification**, and **document summarization**.

### Project Structure
The project is divided into the following main components:

1. **Entity Recognition**: Using the SpaCy library, we extract named entities like people, organizations, dates, and locations from the documents.
2. **Document Classification**: Using **TF-IDF** vectorization and a **Naive Bayes classifier**, we categorize documents into their respective topics.
3. **Document Summarization**: Using **Gensim**, we summarize the content of the documents to extract the most important information.
4. **Visualization**: We visualize the results of the document classification by generating a confusion matrix.

### Dataset
The project uses the **20 Newsgroups** dataset, a widely used dataset in text classification. The dataset contains approximately 18,000 newsgroup posts on 20 different topics. For this project, we use a subset of the dataset focusing on the following categories:
- `rec.autos`
- `rec.sport.baseball`
- `sci.electronics`
- `sci.med`

### Requirements

To run this project, you need the following Python libraries:
- `scikit-learn`
- `spacy`
- `matplotlib`
- `gensim`
- `seaborn`

You can install these dependencies using the following command:

```bash
pip install scikit-learn spacy matplotlib gensim seaborn
```

Additionally, to use SpaCy's **English model**, you need to download it by running the following command:

```bash
python -m spacy download en_core_web_sm
```

### Project Setup

1. Clone this repository or download the project files:
    ```bash
    git clone <repository-url>
    cd document-analysis-nlp
    ```

2. Install the required libraries (as mentioned above).
   
3. Run each script to execute specific parts of the project.

### Code Files

1. **data_preparation.py**:
   - This script loads the 20 Newsgroups dataset, specifies the categories, and prepares the data for analysis.



2. **entity_recognition.py**:
   - This script extracts named entities (persons, organizations, dates, locations) from the news articles using SpaCy.



3. **document_classification.py**:
   - This script performs document classification using **TF-IDF** vectorization and the **Naive Bayes** classifier. The script also calculates the classification accuracy.



4. **summarization.py**:
   - This script generates automatic summaries of the news articles using **Gensim**'s summarization tool.



5. **visualization.py**:
   - This script visualizes the classification results by generating a confusion matrix using **Matplotlib** and **Seaborn**.



### How it Works

1. **Entity Recognition**: 
   - We use **SpaCy** to detect and extract entities such as persons, organizations, dates, and locations from the articles. This helps us identify the key information in each document.
   
2. **Document Classification**: 
   - We convert the documents into numerical vectors using **TF-IDF**. Then, we train a **Naive Bayes classifier** to categorize the documents into different topics (autos, sports, medicine, electronics).
   - After training, the classifier predicts the topics for unseen documents, and we calculate the classification accuracy.

3. **Summarization**:
   - Using **Gensim**, we generate summaries of the articles to highlight the most important information. This is particularly useful when dealing with long articles.

4. **Visualization**:
   - We generate a **confusion matrix** to assess the classifier's performance. This matrix shows how well the model classified each document, helping us visualize where the model performs well and where it struggles.

### Output

**Entity Recognition**:
```
Article 1:
Entities: {'Persons': ['Donald P Boell', 'Pitchers', 'Scott Aldred', 'Andy Ashby', 'Willie Blair', 'Butch Henry', 'Darren Holmes', 'David Neid', 'Jeff Parrett', 'Steve Reed', 'Bruce Ruffin', 'Bryn Smith', 'Gary Wayne', 'Joe Girardi', 'Danny Sheaffer', 'Vinny Castilla', 'Andres Galarraga', 'Charlie Hayes', '.250,48', 'Jim Tatum', 'Eric Young', 'Dante Bichette', 'Daryl Boston', 'Jerald Clark', 'Alex Cole', 'Gerald Young', 'Dale Murphy'], 'Organizations': ['boell@hpcc01.corp.hp.com', 'HP Corporate', 'MLB Totals', 'RBI', '.110', 'RBI', 'Freddie Benavides', 'RBI', 'SS', 'RBI', 'RBI', 'RBI', 'RBI', 'RBI', '.254,38 HR,176 RBI', 'RBI', 'RBI', 'RBI', 'HR,109 RBI,153 SB', 'RBI'], 'Dates': ['.246', '.125', '.283', '.246', '.266', '1259'], 'Locations': ['SB', 'SB', 'SB', 'SB', 'SB']}

Article 2:
Entities: {'Persons': ['Sherri Nichols', 'Young Catchers', 'David M. Tate', 'David', 'Jay Bell', 'Sherri Nichols'], 'Organizations': ['Lopez', 'AAA', 'AAA'], 'Dates': ['22-year old', '23-year old', '24-year-old', 'age 22', 'the last 10 years or so', 'age 21', 'the same season', 'next year', 'age 22', 'the next year', 'age 24'], 'Locations': ['Cleveland']}

Article 3:
Entities: {'Persons': ['Mark Dean', 'Russell Wong', 'Henry\n> Ford', 'Benz', '-Eh', 'Chevy', 'Toyokogio', 'Insert Car', 'Dan Reed - blu@cellar.org - Eat Your Pets - Poke Out'], 'Organizations': ['Ford', 'Organization:', 'Cray Research\t\nLines', 'Ford'], 'Dates': [], 'Locations': ['MD']}

Article 4:
Entities: {'Persons': ['Tod Edward Kurt', 'Tod\n\t\t\t\t\t\t\ttod@cco.caltech.edu'], 'Organizations': ['HP', 'Organization: California Institute of Technology', 'Pasadena\nLines', 'Hewlett Packard'], 'Dates': ['10248B', '1615A', '1615A'], 'Locations': []}

Article 5:
Entities: {'Persons': [], 'Organizations': ['Chicago Home', 'hyundai', 'H.H.M.'], 'Dates': [], 'Locations': []}

```

**Classification Accuracy**:
```
Classification Accuracy: 0.9684873949579832
```
**Confusion Matrix**:

<img width="644" alt="Screenshot 2024-10-05 at 19 26 40" src="https://github.com/user-attachments/assets/bbffbd48-23a3-490c-b707-33f49a3616a6">

