import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time

# Path to your ChromeDriver
chromedriver_path = '/usr/local/bin/chromedriver'

# Initialize ChromeDriver
print("[DEBUG] Setting up ChromeDriver.")
driver = webdriver.Chrome(service=Service(chromedriver_path))

try:
    # Step 1: Go to the login page
    print("[DEBUG] Navigating to the login page.")
    driver.get("https://huawei.agorize.com/en/users/sign_in?redirect_to=https%3A%2F%2Fhuawei.agorize.com%2Fen")
    time.sleep(1)

    # Step 2: Input email and password
    print("[DEBUG] Typing in the email and password.")
    email_input = driver.find_element(By.ID, "user_email")
    email_input.send_keys("")
    password_input = driver.find_element(By.ID, "user_password")
    password_input.send_keys(" ")
    
    # Step 3: Click the login button
    print("[DEBUG] Clicking the login button.")
    login_button = driver.find_element(By.CSS_SELECTOR, 'button[type="submit"]')
    driver.execute_script("arguments[0].click();", login_button)
    time.sleep(2)  # Allow some time for login to process

    # Step 4: Navigate to the participation page after login
    print("[DEBUG] Navigating to the participation page.")
    driver.get("https://huawei.agorize.com/en/challenges/2024-francetecharena/teams/4773/participations")
    time.sleep(2)
    
    # Step 5: Scroll to the bottom of the page
    print("[DEBUG] Scrolling to the bottom of the page.")
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)

    
    try:
        # Step 6: Enable the dropdown
        print("[DEBUG] Finding the dropdown element.")
        dropdown = driver.find_element(By.CLASS_NAME, "dropdown-toggle")
        dropdown.click()
        print("[DEBUG] Dropdown opened.")
        time.sleep(2)
        # Step 7: Select "Remove from my application"
        remove_link = driver.find_element(By.XPATH, "//a[contains(text(), 'Remove from my application')]")
        print("[DEBUG] Finding and clicking 'Remove from my application'.")
        remove_link.click()
        time.sleep(2)

        # Step 8: Confirm the "Are you sure" dialog by clicking OK
        print("[DEBUG] Handling the confirmation alert.")
        alert = driver.switch_to.alert
        alert.accept()
        print("[DEBUG] Confirmation accepted.")
        time.sleep(2)
    except Exception as e:
        print("[DEBUG] 'Remove from my application' not found, proceeding.")
    # Step 9: Wait for the 'Edit Document' buttons and select the last one
    print("[DEBUG] Waiting for the 'Edit Document' buttons to be visible.")
    try:
        # Find all elements that represent 'Edit Document' buttons (assuming they have the same class)
        edit_document_buttons = WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CLASS_NAME, "btn-fpicker"))
        )
        
        # Check if there are at least 4 buttons
        if len(edit_document_buttons) >= 4:
            # Select the 4th (last) button
            last_edit_document_button = edit_document_buttons[-1]  # Or use [3] for the exact 4th one
            print("[DEBUG] Clicking the last 'Edit Document' button.")
            last_edit_document_button.click()  # Click the last one
            time.sleep(2)  # Wait for the next section to load
        else:
            print("[ERROR] Less than 4 'Edit Document' buttons found.")
        
    except Exception as e:
        print(f"[ERROR] Failed to find or interact with the 'Edit Document' buttons: {e}")

    # Step 10: Upload the file using the input field for the last document
    print("[DEBUG] Waiting for the 'Select Files to Upload' input to be visible.")
    try:
        # Wait for the file input for the last document (assuming id remains 'fsp-fileUpload')
        file_input = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "fsp-fileUpload"))
        )
        
        # Upload the file (main.cpp)
        print("[DEBUG] Uploading the main.cpp file for the last document.")
        current_directory = os.getcwd()  # Get the current working directory
        file_path = os.path.join(current_directory, "build/main.cpp")  # Build the full file path
        file_input.send_keys(file_path)  # Send the full file path to the input element
        print("[DEBUG] main.cpp uploaded for the last document.")
        
        # Wait for the upload to complete
        time.sleep(2)
        
    except Exception as e:
        print(f"[ERROR] Failed to find or interact with the file input: {e}")

    # Step 11: Click the upload button to complete the process
    print("[DEBUG] Waiting for the upload button to be clickable.")
    try:
        # Wait for the upload button to be clickable (assuming same selector for the button)
        upload_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "span.fsp-button-upload"))
        )
        
        # Click the upload button
        driver.execute_script("arguments[0].click();", upload_button)
        print("[DEBUG] Upload button clicked for the last document.")
        
        # Wait for the process to finish
        time.sleep(5)
        
    except Exception as e:
        print(f"[ERROR] Failed to find or click the upload button: {e}")

    # Close the driver after completion
    print("[DEBUG] Process completed. Waiting for 5 seconds before closing the driver.")
    time.sleep(5)
    
except Exception as e:
    print(f"[ERROR] An error occurred: {e}")
finally:
    print("[DEBUG] Quitting the driver.")
    driver.quit()