![humAIne logo](https://github.com/e-tony/IvyHacksProject/blob/main/humAIne.png)

# humAIne 

a IvyHacks 2020 project.

## Inspiration

While algorithmic biases and inequalities are worsening by the day, technology and sociology operate in silos. Our survey respondents shared very similar sentiments: developers don’t have the adequate training, regulations, or resources to evaluate the negative societal impacts that their own software have, while sociologists and ethicists often don’t have the technical background to contribute.

To bridge this divide, we wanted to design a solution that connects the two areas of expertise. humAIne provides an accessible platform that educates developers on ethics, and sociologists on AI, allowing them to speak the same language.

![demo](https://github.com/e-tony/IvyHacksProject/blob/main/demo.gif)

## What it does

The first key feature is a free visualizer where a user can select a machine learning model, a debiasing model, and evaluation metrics for any dataset of terms. When a user runs the bias-test, they see a visual representation and a simplified explanation of what these results mean.

The second key feature is an open forum for anyone to discuss the implications of these results or get insights from people outside of their field. We really emphasized the importance of making our platform accessible (even to people who have neither a technology nor a sociology background) but also making sure that users could make informed contributions to the forums. 

So the final feature is a required set of modules; those with technical backgrounds need to complete the sociology module, and those with soc backgrounds need to learn about AI, ML, and NLP. Once they successfully complete these modules, they become “verified” as a contributor who has undergone the basic training from the other side. They can also take additional modules to stack up their knowledge and badges indicating their credibility.

## How we built it
We used Python, Gensim, Plotly, Scikit-learn, Streamlit, FastAPI, ReactJS, wefe, and more to build our app. 

We created a REST API via FastAPI. Our Streamlit app makes calls to the API. We used Plotly to visualize 3D scatter plots and the [Wefe](https://github.com/e-tony/wefe) toolkit to evaluate the model. Additionally, we used a recent debiasing method by [Lauscher et al. (2020)](https://arxiv.org/abs/1909.06092) to show how debiasing can affect the model. 

## Challenges we ran into
The biggest challenge we ran into was deciding on how to connect people with very different backgrounds to collaborate on a very technical topic. 

## What's next for humAIne
We want to add more models, debiasing methods, and metrics to create a "one-stop-shop" for anyone with and without technical knowledge to quickly and easily play around with Machine Learning and contribute to the conversation on Fair AI. 