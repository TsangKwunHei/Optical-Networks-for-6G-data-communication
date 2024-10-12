from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time

# Path to your ChromeDriver
chromedriver_path = '/usr/local/bin/chromedriver'

# Chrome options to connect to an already running Chrome session
chrome_options = Options()
chrome_options.add_experimental_option("debuggerAddress", "localhost:9222")

print("[DEBUG] Setting up ChromeDriver with options.")
# Initialize the driver with the path to ChromeDriver
driver = webdriver.Chrome(service=Service(chromedriver_path), options=chrome_options)

try:
    # Step 1: Go to the specified page
    print("[DEBUG] Navigating to the participation page.")
    driver.get("https://huawei.agorize.com/en/challenges/2024-francetecharena/teams/4773/participations")
    time.sleep(2)
    
    # Step 2: Scroll to the bottom of the page
    print("[DEBUG] Scrolling to the bottom of the page.")
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)
    
    # Step 3: Enable the dropdown
    print("[DEBUG] Finding the dropdown element.")
    dropdown = driver.find_element(By.CLASS_NAME, "dropdown-toggle")
    dropdown.click()
    print("[DEBUG] Dropdown opened.")
    time.sleep(1)
    
    # Step 4: Select "Remove from my application"
    print("[DEBUG] Finding and clicking 'Remove from my application'.")
    remove_link = driver.find_element(By.XPATH, "//a[contains(text(), 'Remove from my application')]")
    remove_link.click()
    time.sleep(1)
    
    # Step 5: Confirm the "Are you sure" dialog by clicking OK
    print("[DEBUG] Handling the confirmation alert.")
    alert = driver.switch_to.alert
    alert.accept()
    print("[DEBUG] Confirmation accepted.")
    time.sleep(2)
    
    # Step 6: Wait for the page to load the "Add a document" box
    print("[DEBUG] Waiting for the 'Add a document' button to be visible.")
    add_document_box = driver.find_element(By.CLASS_NAME, "btn-fpicker")
    add_document_box.click()
    print("[DEBUG] 'Add a document' clicked.")
    time.sleep(2)
    
    # Step 7: Upload the main.cpp file
    print("[DEBUG] Uploading the main.cpp file.")
    file_input = driver.find_element(By.CSS_SELECTOR, "input[type='file']")
    file_input.send_keys("/path/to/main.cpp")
    print("[DEBUG] main.cpp uploaded.")
    
    # Step 8: Press the upload button
    print("[DEBUG] Clicking the upload button.")
    upload_button = driver.find_element(By.XPATH, "//button[text()='Upload']")
    upload_button.click()
    print("[DEBUG] Upload button clicked.")
    
    # Close the driver after completion
    print("[DEBUG] Process completed. Waiting for 5 seconds before closing the driver.")
    time.sleep(5)
    
except Exception as e:
    print(f"[ERROR] An error occurred: {e}")
finally:
    print("[DEBUG] Quitting the driver.")
    driver.quit()