
# Lottery Predictor

This project is a machine learning model designed to predict the most likely set of lottery numbers 6/45 for the next drawing based on previous winning numbers.

## Getting Started

To use this model, you will need to have Python installed on your computer, as well as the following libraries:

-   pandas
-   sklearn
-   numpy
-   tensorflow
-   openpyxl

To install the libraries, run the following command:

Copy code

`pip install numpy pandas tensorflow sklearn openpyxl` 

## Usage

1.  Download the previous winning lottery numbers from your state's lottery website and save them in an Excel file.
2.  2. Run the file `Predictor.py`, which will train the LSTM (Long Short-Term Memory) model to predict the next set of numbers based on historical data and generate a set of predicted numbers.
3.  The program will output the most likely set of numbers for the next drawing.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). You are free to use, modify, and distribute this project as long as you give attribution to the original author.
