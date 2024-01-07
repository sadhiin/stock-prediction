# Machine Learning Model for Stock Prediction

## Objective

The primary goal of this project is to deploy a machine learning model capable of predicting stock market prices. This involves training a predictive model, developing a system for real-time deployment, and providing an API endpoint for clients to access stock predictions.

## Built With

* [Python](https://www.python.org/) - The primary programming language used
* [Flask](https://flask.palletsprojects.com/) - The web framework used for the API
* [Pandas](https://pandas.pydata.org/) - Used for data manipulation and analysis
* [scikit-learn](https://scikit-learn.org/) - Used for machine learning model development



## [Bonus] Monitoring System

Implementing a monitoring system to track the model’s performance and accuracy over time is an additional feature of this project. Tracking the `MAE`, `Rmse`, `R2` score of the model.
Follow the below link to chek the DogsHub reporsitory:

[DagsHub repository](https://dagshub.com/sadhiin/stock-prediction)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

What things you need to install the software and how to install them:
```bash
conda create env
conda activate env
```

```bash
# Example
pip install -r requirements.txt
```

### Installing

A step-by-step series of examples that tell you how to get a development environment running:

```bash
cd stock-prediction
python setup.py install # optional
```

Train the model
```bash
python src/train_eval.py --config=params.yaml
```

Run the we app
```bash
python app.py
```

## Authors

* **Sadhin** - *Initial work* - [GithubProfile](https://github.com/sadhiin)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
