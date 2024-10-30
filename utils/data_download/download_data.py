import argparse
import os
import requests

def download(url, save_name):
    pdf_path = "documents/" + save_name
    if not os.path.exists(pdf_path):
        print("[INFO]: File doesn't exist, downloading...")

        # Send a GET request to the URL
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            with open(pdf_path, "wb") as file:
                file.write(response.content)
                print(f"[INFO]: The file has been downloaded and saved as {pdf_path}")
        else: 
            print(f"[ERROR]: Failed to download the file. Status code: {response.status_code}")
    else:
        print(f"[INFO]: File {pdf_path} exists.")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--document-url", "--url", type=str, help="Document download URL")
    parser.add_argument("--save-name", "--name", type=str, help="Document save name")
    args = parser.parse_args()

    download(args.document_url, args.save_name)