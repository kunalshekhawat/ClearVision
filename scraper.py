import os
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import base64
import time

# Setup selenium Webdriver
def create_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    return driver

#Scroll down to load more images
def scroll_down(driver, scroll_pause_time=2, scroll_limit=5):
    last_height = driver.execute_script("return document.body.scrollHeight")
    for i in range(scroll_limit):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(scroll_pause_time)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

#Scrape all images
def scrape_all_images(driver):
    try:
        images = driver.find_elements(By.TAG_NAME, 'img')
        image_urls = []
        for img in images:
            image_url = img.get_attribute('src') or img.get_attribute('data-src')
            if image_url and "data:image/gif" not in image_url:
                width = int(img.get_attribute('width') or 0)
                height = int(img.get_attribute('height') or 0)
                if width >= 100 and height >= 100:
                    image_urls.append(image_url)
        return image_urls
    except Exception as e:
        print(f"Error scraping images: {e}")
        return []

#Save the image function
def save_image(image_url, folder_name, file_name, retry_count=3):
    try:
        file_path =os.path.join(folder_name, f"{file_name}.jpg")

        if image_url.startswith('data:image/'):
            header, encoded = image_url.split(',', 1)
            image_data = base64.b64decode(encoded)
            with open(file_path, 'wb') as f:
                f.write(image_data)

        else:
            for attempt in range(retry_count):
                response = requests.get(image_url, timeout=10)
                if response.status_code == 200:
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                    break
                else:
                    print(f"Failed attempt {attempt+1} for image: {image_url}")
                    time.sleep(2)
    except Exception as e:
        print(f"Error saving image {file_name}: {e}")

#Main scraping function
def scrape_and_save_images(base_folder, num_images=1000):
    print("\n--- Starting to scrape and save ---")
    print(f"1. The base folder passed to this function is: {base_folder}")

    # Define the target subfolder name
    subfolder_name = "scraped_images" # Or "scraped_plants", your choice
    folder_path = os.path.join(base_folder, subfolder_name)
    print(f"2. The full path for the new folder should be: {folder_path}")

    try:
        # This is the critical step
        if not os.path.exists(folder_path):
            print("3. Folder does not exist. Attempting to create it now...")
            os.makedirs(folder_path)
            print(f"✅ SUCCESS: Folder '{folder_path}' was created.")
        else:
            print(f"ℹ️ INFO: Folder '{folder_path}' already exists. No action needed.")
    except Exception as e:
        # If there is ANY error during folder creation, this will catch it and print it.
        print(f"\n❌ FAILED TO CREATE FOLDER. Error: {e}\n")
        # Exit the function because we can't save images without the folder.
        return

    # If the script reaches this point, folder creation was successful.
    print("4. Proceeding to scrape images...")
    
    driver = create_driver()
    search_term = "high resolution potraits"
    driver.get(f"https://www.google.com/search?q={search_term}&tbm=isch")
    time.sleep(5)
    
    scroll_down(driver)
    image_urls = scrape_all_images(driver)
    
    if not image_urls:
        print("⚠️ WARNING: No image URLs were found after scraping.")
    
    image_urls = image_urls[:num_images]
    for index, image_url in enumerate(image_urls, start=1):
        file_name = f'{search_term}_{index}'
        save_image(image_url, folder_path, file_name)
        
    print(f"Finished scraping function.")
    driver.quit()

if __name__ == "__main__":
    # 1. Find the user's home directory (e.g., C:\Users\YourName)
    user_profile = os.path.expanduser("~")

    # 2. Join it with "Desktop" and your target folder "ClearVision"
    # This creates the correct, full path
    base_folder = os.path.join(user_profile,"OneDrive" ,"Desktop", "ClearVision")

    print(f"Saving images to: {base_folder}")
    scrape_and_save_images(base_folder, num_images=1000)