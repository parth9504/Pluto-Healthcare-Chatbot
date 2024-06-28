# Pluto-Healthcare-Chatbot
A healthcare chatbot which aims to **analyze symptoms**, provide **information about diseases** and **suggest medicines for treatment**. 

This chatbot has been specifically built for analyzing symptoms of various diseases,
extracting and accessing valuable information about diseases, including signs &
symptoms, treatments, precautions, and other medical care needed, along with
accessing medicinal information.
The idea here is to provide the users with a relatively faster conversation experience
where they wouldn’t have to spend a lot of time on typing messages for the chatbot,
rather would get the things done by just selecting an option and only providing the
relevant information.
I would like to mention here that for this project, I haven’t used specific healthcare
APIs or datasets like MIMIC-III. As a result, it might not appear as a
very optimal and efficient way of designing a chatbot, especially for a field like
healthcare where accurate predictions play a crucial role. However, I have designed
this project to get started and explore the ways of creating chatbots from scratch,
using traditional machine learning algorithms along with natural language processing
and other Python-based libraries.

**Project Overview**

The chatbot built here provides three options for the user to choose from:
1. **Analyze Symptoms**: The user can provide the symptoms of their health
condition, and the input is then used in the model to predict the disease they
might be suffering from.
The dataset used for this model includes conversations between doctors and
patients along with the disease label as part of the diagnosis. Natural
Language Processing (NLP) was used to preprocess the data, and essential
keywords were extracted from the conversations. The snapshot below shows
the word cloud with the most commonly occurring words in the conversations
of the dataset.

![Screenshot (410)](https://github.com/parth9504/Pluto-Healthcare-Chatbot/assets/127659489/9f891b3d-3b39-484a-88e9-8699c16992b1)


For the model, a voting mechanism technique was used. The idea was to increase
the overall accuracy of the model. To achieve this, a group of classifiers like Logistic
Regression, Support Vector Machine, Multinomial Naive Bayes, and Random Forest
Classifier were used. When an input is passed to the model, all these classifiers
work concurrently to predict the best possible disease, and the majority of the votes
decide the prediction of the disease. The accuracy achieved by the trained model
was 98.3%.

![Screenshot (409)](https://github.com/parth9504/Pluto-Healthcare-Chatbot/assets/127659489/b1a58e67-feaf-4054-9744-e8a78a2b49af)

2) **Access Information about Diseases**: This section allows users to access
information about any disease. For this, the Wikipedia module available in Python
was used. **Wikipedia Module** is a Python library that makes it easy to access and parse data
from Wikipedia. This provides a very convenient way to view specific sections of the
Wikipedia page, search for articles, get summaries, and fetch other details about
Wikipedia pages programmatically. When the user enters a search query (name of a
disease or a symptom), the possible search results are displayed, and post the
selections, the available sections on the Wikipedia page are shown, making it easier
for the user to access information quickly with just a click of a button.

**More information about the Python module:** [Wikipedia package on PyPI](https://pypi.org/project/wikipedia/)


3) **Medicinal Information**: This section provides users with medicinal data, where
they can enter the name of any disease and the best-matched medicines available in
the market or pharmacies can be shown along with their composition and side
effects. The dataset used here was available on Kaggle (11000 Medicine Details).
To facilitate the working of this section, TF-IDF vectorizer was used along with the
Nearest Neighbors model (metric: cosine similarity). The Nearest Neighbors model is
based on the idea that similar instances are close to each other in the feature space.
It is a non-parametric method, meaning it makes no assumptions about the
underlying data distribution. The model is often referred to as K-Nearest Neighbors
(K-NN) when used for classification. The model was used on a dataset containing
11,000 medicinal information entries provided by the renowned online pharmacy
1mg.


The GUI of the chatbot was built using Streamlit, which provides multiple features to
create chatbots conveniently.
These chat elements are designed to be used in conjunction with each other, but
they can also be used separately.

**You can refer to this link to read the documentation of Streamlit:** [Streamlit](https://docs.streamlit.io/develop/api-reference/chat)

**The screenshots attached below demonstrate the working of the chatbot**

![Screenshot (418)](https://github.com/parth9504/Pluto-Healthcare-Chatbot/assets/127659489/b39b200c-19ee-4084-9daf-2d6270053325)


![Screenshot (419)](https://github.com/parth9504/Pluto-Healthcare-Chatbot/assets/127659489/1d99cdf7-0f44-4bca-b52c-2302888ca256)


![Screenshot (420)](https://github.com/parth9504/Pluto-Healthcare-Chatbot/assets/127659489/d86d2697-bf23-4371-aeb3-5e5291e62c92)


![Screenshot (421)](https://github.com/parth9504/Pluto-Healthcare-Chatbot/assets/127659489/600e158f-f1ea-44e2-be20-b8a341b8b1f5)


![Screenshot (422)](https://github.com/parth9504/Pluto-Healthcare-Chatbot/assets/127659489/4b64d791-8a50-4884-b5de-761a39f16c4f)


![Screenshot (423)](https://github.com/parth9504/Pluto-Healthcare-Chatbot/assets/127659489/f8b19e33-dc69-4f91-a7f3-ffd06a098e28)

