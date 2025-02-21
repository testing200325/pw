import argparse
import requests
import machine_learning as ml
import feature_extraction as fe
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt


def fetch_website_content(url):
    """Fetches the HTML content of a given URL."""
    try:
        response = requests.get(url, verify=True, timeout=4)
        if response.status_code != 200:
            print(
                f"HTTP connection failed for {url} (Status Code: {response.status_code})")
            return None
        return response.content
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
        return None


def detect_phishing(url, model):
    """Detects if a given URL is phishing using the specified ML model."""
    html_content = fetch_website_content(url)
    if html_content is None:
        return

    soup = BeautifulSoup(html_content, "html.parser")
    vector = [fe.create_vector(soup)]  # Convert to 2D array
    result = model.predict(vector)

    if result[0] == 0:
        print("Legitimate website ✅")
    else:
        print("Potential PHISHING website ⚠️")


def show_dataset_info():
    """Displays dataset information and statistics."""
    phishing_count = ml.phishing_df.shape[0]
    legitimate_count = ml.legitimate_df.shape[0]
    total_count = phishing_count + legitimate_count

    phishing_rate = (phishing_count / total_count) * 100
    legitimate_rate = 100 - phishing_rate

    print(f"Dataset Information:")
    print(f"- Phishing websites: {phishing_count}")
    print(f"- Legitimate websites: {legitimate_count}")
    print(f"- Total: {total_count}")

    labels = ["Phishing", "Legitimate"]
    sizes = [phishing_rate, legitimate_rate]
    explode = (0.1, 0)
    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels,
           autopct='%1.1f%%', shadow=True, startangle=90)
    ax.axis('equal')
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Phishing Website Detection using Machine Learning")
    parser.add_argument("url", type=str, help="URL to check")
    parser.add_argument("--model", type=str, choices=[
        "GNB", "SVM", "DT", "RF", "AB", "NN", "KN"
    ], default="GNB", help="Select a machine learning model")
    parser.add_argument("--show-dataset", action="store_true",
                        help="Show dataset statistics")
    args = parser.parse_args()

    model_mapping = {
        "GNB": ml.nb_model,
        "SVM": ml.svm_model,
        "DT": ml.dt_model,
        "RF": ml.rf_model,
        "AB": ml.ab_model,
        "NN": ml.nn_model,
        "KN": ml.kn_model
    }
    selected_model = model_mapping[args.model]

    if args.show_dataset:
        show_dataset_info()
    else:
        detect_phishing(args.url, selected_model)


if __name__ == "__main__":
    main()
