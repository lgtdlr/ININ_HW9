from matplotlib import pyplot as plt
from scipy import stats
import numpy as np
import seaborn
import math
import xlrd

'''
RESOURCES
https://towardsdatascience.com/a-complete-guide-to-confidence-interval-and-examples-in-python-ff417c5cb593
https://web.archive.org/web/20191113020101/https://medium.com/@rrfd/testing-for-normality-applications-with-python-6bf06ed646a9
https://towardsdatascience.com/6-ways-to-test-for-a-normal-distribution-which-one-to-use-9dcf47d8fa93
'''


def parse_sheet(file):
    title = ""
    data = []
    result = []

    # Open Workbook
    wb = xlrd.open_workbook(file)
    xl_sheet = wb.sheet_by_index(0)
    column = xl_sheet.col_values(1)
    first = True

    for cell in column:
        if first:
            title = cell
            first = False
        else:
            data.append(cell)
    result.append(title)
    result.append(data)
    return result


def generate_histogram(title, data):
    bins = []
    SIZE_OF_BINS = 5

    # Create bins based on data
    for i in range(math.floor(min(data)), math.ceil(max(data)), SIZE_OF_BINS):
        bins.append(i)

    seaborn.histplot(data=data, bins=bins)

    # Alternative methods
    # seaborn.histplot(data=data, bins=bins)
    # plt.hist(data, bins=bins)

    plt.title("Histogram of " + title)
    plt.show()


def generate_box_and_whisker(title, data):
    seaborn.boxplot(data=data)
    # plt.boxplot(data)
    plt.title("Boxplot of " + title)
    plt.show()


def generate_probability_plot(title, data):
    stats.probplot(data, dist="norm", plot=plt)
    plt.title("Probability Plot of " + title)
    plt.show()


def verify_normality(title, data):
    print("\nRESULTS\n")
    # Shapiro-Wilk test
    # Tests if sample came from a normal dist
    shapiroResult = stats.shapiro(data)
    print(shapiroResult)

    # Anderson-Darling test
    # Tests if data comes from a particular distribution (normal in this case)
    # Not sure how this works yet...
    andersonResult = stats.anderson(data, dist="norm")
    print(andersonResult)

    # Looks useful but haven't seen the documentation to know what it actually calculates
    print(str(stats.normaltest(values)))

    print("\nNull hypothesis : 'Data is normally distributed'.")
    if shapiroResult[1] < 0.05:
        print("Shapiro-Wilk Test: Null Rejected due to p-value < 0.05" + " (" + str(
            shapiroResult[1]) + ")" + ". Sample does NOT come from a normal distribution.")
        return False
    else:
        print(
            "Shapiro-Wilk Test: Null Accepted. Failed to reject null hypothesis. Sample possible comes from a normal "
            "distribution.")
        return True


def calculate_lower_confidence_interval(values, mean, confidence):
    # DoF (Sample size - 1)
    degreesOfFreedom = len(values) - 1
    alpha = 1 - confidence
    sampleMean = np.mean(values)
    standardError = stats.sem(values)

    # Since what is wanted is lower confidence, and the stats.t.interval function calculates a two-tailed confidence
    # interval, the confidence is adjusted to work.
    adjustedConfidence = confidence - alpha

    confidenceInterval = stats.t.interval(adjustedConfidence, degreesOfFreedom, sampleMean, standardError)
    print("\nConfidence Interval : " + str(confidenceInterval))
    if mean < confidenceInterval[0]:
        print(str(confidence * 100) + "% confident that the mean is greater than " + str(mean) + " seconds.")
    else:
        print(str(confidence * 100) + "% confident that the mean NOT is greater than " + str(mean) + " seconds.")


def calculate_confidence_interval_samples(values, mean, confidence):
    # DoF (Sample size - 1)
    degreesOfFreedom = len(values) - 1
    alpha = 1 - confidence
    sampleMean = np.mean(values)
    standardError = stats.sem(values)

    confidenceInterval = stats.t.interval(confidence, degreesOfFreedom, sampleMean, standardError)
    print("\nConfidence Interval : " + str(confidenceInterval))
    if confidenceInterval[0] < mean < confidenceInterval[1]:
        print("Mean finish time for the process is greater than " + str(mean) + " seconds with a " + str(
            confidence * 100) + " confidence level.")


def calculate_confidence_interval_summarized_data(count, xBar, stdDev, alpha):
    result = []
    Z = stats.norm.ppf(alpha / 2)

    result.append(xBar - (Z * (stdDev / np.sqrt(count))))
    result.append(xBar + (Z * (stdDev / np.sqrt(count))))

    print("\nConfidence Interval: " + str(result))


if __name__ == '__main__':
    # Location of the file
    loc = "./HW8_Data.xlsx"

    data = parse_sheet(loc)
    title = data[0]
    values = data[1]

    # Problem 2b
    print("PROBLEM 2b")
    calculate_confidence_interval_summarized_data(20, 5.23, 0.24, 0.05)

    # Problem 3a
    print("PROBLEM 3a")
    # IMPRESSIONS
    #   1. Many outliers in the boxplot which is representative that it is not normally distributed.
    generate_box_and_whisker(title, values)

    # Problem 3b
    print("PROBLEM 3b")
    # IMPRESSIONS
    #   1. Sort of looks normally distributed but considering the sample size of 1000, it is unlikely that at that size
    #   the histogram would be spiked in the middle as it is. One would expect a smoother bell curve.
    generate_histogram(title, values)

    # Problem 3c
    print("PROBLEM 3c")
    generate_probability_plot(title, values)
    verify_normality(title, values)

    # Problem 3d
    print("PROBLEM 3d")
    calculate_lower_confidence_interval(values, 800, 0.99)

    # Problem 3e
    print("PROBLEM 3e")
    calculate_confidence_interval_samples(values, 800, 0.95)
